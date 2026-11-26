"""Chapter publication metadata, independent of the TTS runtime and Gradio."""

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


def positive_version(value: int | float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("La version debe ser un entero positivo.")
    if not math.isfinite(value) or value < 1 or int(value) != value:
        raise ValueError("La version debe ser un entero positivo.")
    return int(value)


def validate_gap(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("gap_sec debe ser un numero finito mayor o igual que cero.")
    if not math.isfinite(value) or value < 0:
        raise ValueError("gap_sec debe ser un numero finito mayor o igual que cero.")
    return float(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scripts_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256(b"chatterbox-scripts-v1\0")
    for path in sorted(paths, key=lambda source: source.name):
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        for value in (name, content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def reference_voice_id(path: str | None) -> str:
    if not path or not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError(
            "Falta la voz: selecciona o sube un audio de referencia local no vacio en Gradio."
        )
    return f"reference-sha256:{sha256_file(Path(path))}"


def request_hash(configuration: dict) -> str:
    payload = json.dumps(
        configuration, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def chapter_prefix(podcast: str, language: str, chapter: int, version: int) -> str:
    if not podcast or podcast in (".", "..") or re.search(r"[/\\:]", podcast):
        raise ValueError("El identificador de podcast debe ser un unico componente de ruta.")
    if not re.fullmatch(r"[a-z]{2}", language):
        raise ValueError("El idioma debe ser un codigo de dos letras minusculas.")
    chapter = positive_version(chapter)
    version = positive_version(version)
    return f"{podcast}/{language}/ch{chapter:02d}/v{version}"


def quiz_question_numbers(tags: set[str]) -> set[str]:
    shared = {"PREGUNTAS", "INCORRECTO", "CORRECTO", "ESPERA_1", "ESPERA_2", "FIN_PREGUNTAS"}
    tags = tags - {"CONTINUAR"}  # opcional
    questions = {
        tag.removeprefix("PREGUNTA_")
        for tag in tags if re.fullmatch(r"PREGUNTA_[1-9]\d*", tag)
    }
    if not questions:
        raise ValueError("El quiz debe contener al menos una PREGUNTA_N.")
    expected = shared | {
        f"{prefix}_{number}"
        for number in questions for prefix in ("PREGUNTA", "OPCION_CORRECTA", "RESPUESTA")
    }
    if tags != expected:
        raise ValueError(
            f"Quiz incompleto o etiquetas sin pregunta: {', '.join(sorted(tags ^ expected))}"
        )
    return questions


def quiz_metadata(blocks: list[dict]) -> list[dict]:
    if not blocks:
        return []
    by_tag = {str(block["tag"]): str(block["text"]).strip() for block in blocks}
    if len(by_tag) != len(blocks) or any(not text for text in by_tag.values()):
        raise ValueError("Las etiquetas de quiz deben ser unicas y tener texto.")
    questions = quiz_question_numbers(set(by_tag))
    metadata = {tag: {"tag": tag} for tag in by_tag}
    option_pattern = re.compile(r"(?<!\w)(?:\(([A-Za-z])\)|([A-Za-z])[.):])\s+")
    for number in questions:
        tag = f"PREGUNTA_{number}"
        matches = list(option_pattern.finditer(by_tag[tag]))
        letters = [(match.group(1) or match.group(2)).upper() for match in matches]
        if len(letters) < 2 or len(set(letters)) != len(letters):
            raise ValueError(f"[{tag}]: identifica al menos dos alternativas distintas con A., B., A) o (A).")
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(by_tag[tag])
            if not by_tag[tag][match.end():end].strip():
                raise ValueError(f"[{tag}]: hay una alternativa sin texto.")
        correct_tag = f"OPCION_CORRECTA_{number}"
        correct = by_tag[correct_tag].upper()
        if not re.fullmatch(r"[A-Z]", correct) or correct not in letters:
            raise ValueError(f"[{correct_tag}] debe contener solo una letra incluida en [{tag}].")
        metadata[tag]["option_letters"] = letters
        metadata[correct_tag]["correct_option"] = correct
    return list(metadata.values())


def mp3_metadata(path: Path) -> dict:
    from mutagen import MutagenError
    from mutagen.mp3 import MP3

    try:
        info = MP3(path).info
    except MutagenError as error:
        raise ValueError(f"MP3 final no valido: {path.name}: {error}") from error
    return {
        "duration_sec": info.length,
        "bitrate_kbps": info.bitrate / 1000,
        "sha256": sha256_file(path),
    }


def require_fields(data: dict, fields: Sequence[str], context: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{context} debe ser un objeto.")
    missing = set(fields) - data.keys()
    if missing:
        raise ValueError(f"Faltan datos obligatorios en {context}: {', '.join(sorted(missing))}.")


def validate_chapter_manifest(manifest: dict) -> None:
    require_fields(
        manifest, ("schema_version", "podcast", "language", "chapter", "title", "audio", "blocks", "quiz", "generated"),
        "manifest",
    )
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("schema_version debe ser 1.")
    audio = manifest["audio"]
    require_fields(audio, ("r2_key", "duration_sec", "bitrate_kbps", "sha256"), "audio")
    if type(manifest["chapter"]) is not int:
        raise ValueError("chapter debe ser un entero positivo.")
    key = audio["r2_key"]
    version_match = re.search(r"/v([1-9]\d*)/ch\d+\.mp3$", key)
    if not version_match:
        raise ValueError("El audio debe tener una ruta versionada vN.")
    prefix = chapter_prefix(
        manifest["podcast"], manifest["language"], manifest["chapter"], int(version_match[1]),
    )
    if key != f"{prefix}/ch{manifest['chapter']:02d}.mp3":
        raise ValueError("La ruta del MP3 no corresponde al capitulo.")
    if not isinstance(manifest.get("title"), str) or not manifest["title"].strip():
        raise ValueError("Falta el titulo real del capitulo en [NOMBRE_CAPITULO].")
    for field in ("duration_sec", "bitrate_kbps"):
        value = audio[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"audio.{field} debe ser un numero finito positivo.")
    generated = manifest["generated"]
    require_fields(
        generated, ("date", "tts", "model", "voice", "gap_sec", "scripts_hash", "podcast_mix_version"),
        "generated",
    )
    for field in ("tts", "model", "voice"):
        value = generated.get(field)
        if not isinstance(value, str) or not value.strip() or value.lower() in ("default", "unknown"):
            raise ValueError(f"Falta configuracion obligatoria: generated.{field}.")
    validate_gap(generated["gap_sec"])
    positive_version(generated["podcast_mix_version"])
    hashes = [audio["sha256"], generated["scripts_hash"]]
    if "request_hash" in generated:
        hashes.append(generated["request_hash"])
    for value in hashes:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Los hashes deben ser SHA-256 completos en hexadecimal minusculo.")
    date = datetime.fromisoformat(generated["date"].replace("Z", "+00:00"))
    if date.utcoffset() is None or date.utcoffset().total_seconds() != 0:
        raise ValueError("generated.date debe ser una fecha ISO 8601 UTC.")
    blocks = manifest["blocks"]
    if not blocks:
        raise ValueError("El capitulo debe contener bloques de teoria.")
    previous_end = 0
    for order, block in enumerate(blocks, start=1):
        require_fields(block, ("order", "title", "script", "start_sec", "end_sec"), "blocks")
        if type(block["order"]) is not int or block["order"] != order:
            raise ValueError("Los bloques deben estar ordenados consecutivamente desde 1.")
        start, end = block["start_sec"], block["end_sec"]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (start, end)):
            raise ValueError("Los tiempos de bloque deben ser numeros finitos.")
        if not 0 <= start < end <= audio["duration_sec"]:
            raise ValueError(f"Tiempos invalidos en el bloque {order}.")
        if start < previous_end:
            raise ValueError(f"Solapamiento o desorden cronologico en el bloque {order}.")
        previous_end = end
        script = block["script"]
        if not isinstance(script, str) or not script or re.search(r"[/\\:]", script) or script in (".", ".."):
            raise ValueError("script debe ser un nombre de archivo fuente, no una ruta local.")
    quiz = manifest["quiz"]
    tags = {entry["tag"] for entry in quiz}
    if len(tags) != len(quiz):
        raise ValueError("Etiquetas de quiz repetidas en el manifest.")
    if quiz:
        quiz_question_numbers(tags)
    by_tag = {entry["tag"]: entry for entry in quiz}
    for entry in quiz:
        tag = entry["tag"]
        if entry["r2_key"] != f"{prefix}/quiz/{tag.lower()}.mp3":
            raise ValueError(f"Ruta de quiz no valida: {tag}.")
        if re.fullmatch(r"PREGUNTA_[1-9]\d*", tag):
            letters = entry.get("option_letters", [])
            if len(letters) < 2 or len(set(letters)) != len(letters) or any(not re.fullmatch("[A-Z]", letter) for letter in letters):
                raise ValueError(f"option_letters no valido: {tag}.")
            correct = by_tag.get(tag.replace("PREGUNTA_", "OPCION_CORRECTA_"), {}).get("correct_option")
            if correct not in letters:
                raise ValueError(f"correct_option no coincide con option_letters: {tag}.")


def build_chapter_manifest(
    *, podcast: str, language: str, chapter: int, version: int, title: str,
    mp3_path: Path, blocks: list[dict], quiz: list[dict], generated: dict,
) -> dict:
    prefix = chapter_prefix(podcast, language, chapter, version)
    manifest = {
        "schema_version": 1,
        "podcast": podcast,
        "language": language,
        "chapter": chapter,
        "title": title,
        "audio": {"r2_key": f"{prefix}/ch{chapter:02d}.mp3", **mp3_metadata(mp3_path)},
        "blocks": sorted(blocks, key=lambda block: block["order"]),
        "quiz": [
            {**entry, "r2_key": f"{prefix}/quiz/{entry['tag'].lower()}.mp3"}
            for entry in quiz
        ],
        "generated": {**generated, "date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
    }
    validate_chapter_manifest(manifest)
    return manifest
