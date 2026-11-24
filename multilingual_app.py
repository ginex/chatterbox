import random
import gc
import re
import time
import hashlib
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
import lameenc
import numpy as np
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, MULTILINGUAL_T3_MODELS
from chatterbox.tts import ChatterboxTTS
import gradio as gr
from podcast_manifest import (
    build_chapter_manifest,
    chapter_prefix,
    mp3_metadata,
    positive_version,
    quiz_metadata,
    reference_voice_id,
    request_hash,
    scripts_hash,
    sha256_file,
    validate_chapter_manifest,
    validate_gap,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UI_LANGUAGES = {
    "es": "Español",
    "en": "Inglés",
    "fr": "Francés",
}
LANGUAGE_MODELS = {
    "es": {"key": "es-es", "name": "Español (España)"},
    "en": {"key": "en", "name": "Chatterbox English"},
    "fr": {"key": "v3", "name": "Multilingual V3"},
}
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None
LOADED_MODEL_KEY = None
APP_DATA_DIR = Path(__file__).resolve().parent / "chatterbox_output"
CACHE_DIR = APP_DATA_DIR / "cache"
# Mirrors the R2 key layout; publication is local, with no R2 uploads.
PUBLISH_DIR = APP_DATA_DIR / "publish"
REFERENCE_DIR = APP_DATA_DIR / "reference"
PODCAST_MUSIC_DIR = Path(__file__).resolve().parent / "podcast_music"
LAST_REQUEST_PATH = APP_DATA_DIR / "last_request.json"
APP_DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
PUBLISH_DIR.mkdir(exist_ok=True)
REFERENCE_DIR.mkdir(exist_ok=True)
PODCAST_MUSIC_DIR.mkdir(exist_ok=True)

PODCAST_BLOCK_TAGS = (
    "NOMBRE_PODCAST",
    "NOMBRE_CAPITULO",
    "INTRO_CAPITULO",
    "SECCION",
    "FINAL",
)
PODCAST_BLOCK_PATTERN = re.compile(
    rf"\[({'|'.join(PODCAST_BLOCK_TAGS)}|PREGUNTAS|PREGUNTA_[1-9]\d*|"
    r"OPCION_CORRECTA_[1-9]\d*|RESPUESTA_[1-9]\d*|INCORRECTO|CORRECTO|"
    r"ESPERA_[12]|FIN_CAPITULO)\]",
    flags=re.IGNORECASE,
)
PODCAST_MIX_VERSION = 5
FINAL_JINGLE_FILENAME = "jingle2.wav"
# {podcast}__{idioma}__c{NN}[-b{NN}]__{titulo-en-slug}.txt
SCRIPT_NAME_PATTERN = re.compile(
    r"^(?P<podcast>.+?)__(?P<language>[a-z]{2})__c(?P<chapter>\d+)(?:-b(?P<block>\d+))?__(?P<slug>.+)$",
    flags=re.IGNORECASE,
)
PROCESSED_SCRIPTS_DIRNAME = "processed"
INTRO_TAGS = ("NOMBRE_PODCAST", "NOMBRE_CAPITULO", "INTRO_CAPITULO")
MP3_BITRATE_KBPS = 192
VOICE_TARGET_DBFS = -20.0

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب."
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal."
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal."
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube."
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel."
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/es-es/es_es_f1.wav",
        "text": "Hola, ¿cómo estás? Hoy es un día perfecto para dar un paseo por la Plaza Mayor."
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme."
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube."
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו."
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।"
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube."
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。"
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다."
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami."
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal."
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår."
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube."
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube."
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале."
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal."
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube."
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık."
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑. 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。"
    },
}

# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    saved_reference = get_saved_reference_path()
    if saved_reference:
        return saved_reference
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    last_request = load_json(LAST_REQUEST_PATH)
    if last_request and last_request.get("text"):
        return last_request["text"]
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def last_request_value(name: str, default):
    last_request = load_json(LAST_REQUEST_PATH)
    return last_request.get(name, default) if last_request else default


def load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def get_saved_reference_path() -> str | None:
    reference_state = load_json(REFERENCE_DIR / "current.json")
    if not reference_state:
        return None
    reference_path = Path(reference_state.get("path", ""))
    return str(reference_path) if reference_path.is_file() else None


def get_latest_combined_path() -> str | None:
    combined_files = list(CACHE_DIR.glob("*/combined.wav"))
    if not combined_files:
        return None
    latest_file = max(combined_files, key=lambda path: path.stat().st_mtime)
    return str(latest_file.resolve())


def persist_reference_audio(reference: str | None) -> str | None:
    if not reference or not Path(reference).is_file():
        return reference

    source = Path(reference)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    suffix = source.suffix.lower() or ".wav"
    destination = REFERENCE_DIR / f"reference-{digest}{suffix}"
    if source.resolve() != destination.resolve() and not destination.exists():
        shutil.copy2(source, destination)
    save_json(REFERENCE_DIR / "current.json", {"path": str(destination.resolve())})
    return str(destination.resolve())


def get_or_load_model(language_id: str = "es"):
    """Load the model selected for a language, reusing Hugging Face's disk cache."""
    global MODEL, LOADED_MODEL_KEY
    model_spec = LANGUAGE_MODELS.get(language_id)
    if model_spec is None:
        raise ValueError(f"Unsupported UI language: {language_id}")

    model_key = model_spec["key"]
    if MODEL is not None and LOADED_MODEL_KEY == model_key:
        return MODEL

    if MODEL is not None:
        print(f"Unloading model {LOADED_MODEL_KEY} before loading {model_key}...")
        MODEL = None
        LOADED_MODEL_KEY = None
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"Loading model {model_key} for {UI_LANGUAGES[language_id]}...")
    try:
        if model_key == "en":
            MODEL = ChatterboxTTS.from_pretrained(DEVICE)
        else:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(
                DEVICE,
                t3_model=model_key,
            )
        LOADED_MODEL_KEY = model_key
        if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
            MODEL.to(DEVICE)
        print(f"Model {model_key} loaded. Internal device: {getattr(MODEL, 'device', 'N/A')}")
    except Exception as e:
        MODEL = None
        LOADED_MODEL_KEY = None
        print(f"Error loading model {model_key}: {e}")
        raise
    return MODEL

# Load the original Spanish model at startup.
try:
    get_or_load_model("es")
except Exception as e:
    print(f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}")

def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")


NON_TERMINAL_ABBREVIATIONS = frozenset({
    "sr", "sra", "srta", "sres", "dr", "dra", "lic", "ing", "prof", "arq",
    "ud", "uds", "vd", "vds", "dña", "sto", "sta", "ee", "ej",
    "núm", "nº", "pág", "págs", "art", "cap", "vol", "aprox", "av", "avda",
    "vs", "mr", "mrs", "ms", "st", "e.g", "i.e", "mme", "mlle",
})


def ends_with_abbreviation(sentence: str) -> bool:
    last_word = sentence.rsplit(maxsplit=1)[-1].lstrip("([¿¡\"'«")
    if not last_word.endswith("."):
        return False
    token = last_word[:-1].lower()
    # Single letters cover initials (J. K. Rowling) and "p. ej.".
    return token in NON_TERMINAL_ABBREVIATIONS or (len(token) == 1 and token.isalpha())


def split_text_into_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Split text near max_chars, preferring sentence and clause boundaries."""
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    sentences = []
    for piece in re.split(r"(?<=[.!?。！？])\s+", normalized_text):
        if sentences and ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(sentence) <= max_chars:
            candidate = f"{current_chunk} {sentence}".strip()
            if len(candidate) <= max_chars:
                current_chunk = candidate
                continue
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            continue

        if current_chunk:
            chunks.append(current_chunk)
            current_chunk = ""

        clauses = re.split(r"(?<=[,;:])\s+|(?<=[，；：])", sentence)
        for clause in clauses:
            words = clause.split()
            while words:
                part = ""
                while words and len(f"{part} {words[0]}".strip()) <= max_chars:
                    part = f"{part} {words.pop(0)}".strip()
                if not part:
                    part = words.pop(0)
                candidate = f"{current_chunk} {part}".strip()
                if current_chunk and len(candidate) > max_chars:
                    chunks.append(current_chunk)
                    current_chunk = part
                else:
                    current_chunk = candidate

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def parse_podcast_blocks(text: str) -> list[dict[str, str | int]]:
    """Split tagged podcast text into ordered blocks without speaking the tags."""
    matches = list(PODCAST_BLOCK_PATTERN.finditer(text))
    if not matches:
        return []

    blocks = []
    tag_counts = {}
    for index, match in enumerate(matches):
        tag = match.group(1).upper()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():content_end].strip()
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        blocks.append({
            "tag": tag,
            "occurrence": tag_counts[tag],
            "text": content,
        })
    return blocks


def podcast_block_filename(block: dict[str, str | int], index: int) -> str:
    tag = str(block["tag"]).lower()
    occurrence = int(block["occurrence"])
    return f"block-{index:02d}-{tag}-{occurrence:02d}.wav"


def split_podcast_blocks(
    blocks: list[dict[str, str | int]],
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """Keep the quiz out of the theory mix and reject ambiguous quiz filenames."""
    theory = []
    quiz = []
    quiz_tags = set()
    for block in blocks:
        tag = str(block["tag"])
        if tag in PODCAST_BLOCK_TAGS:
            if quiz:
                raise gr.Error(f"El bloque [{tag}] no puede aparecer después del quiz.")
            theory.append(block)
        else:
            if not theory or theory[-1]["tag"] != "FINAL":
                raise gr.Error(f"El bloque [{tag}] debe aparecer después de [FINAL].")
            if tag in quiz_tags:
                raise gr.Error(f"Etiqueta de quiz repetida: [{tag}].")
            quiz_tags.add(tag)
            quiz.append(block)
    return theory, quiz


def podcast_background_filename(block: dict[str, str | int]) -> str | None:
    tag = block["tag"]
    if tag == "INTRO_CAPITULO":
        return "background_music_1.wav"
    if tag == "SECCION":
        section_backgrounds = (
            "background_music_2.wav",
            "background_music_3.wav",
            "background_music_4.wav",
            "background_music_1.wav",
        )
        return section_backgrounds[(int(block["occurrence"]) - 1) % len(section_backgrounds)]
    if tag == "FINAL":
        return "final_music.wav"
    return None


def podcast_transition_filename(
    current_block: dict[str, str | int],
    next_block: dict[str, str | int],
) -> str | None:
    transition = (current_block["tag"], next_block["tag"])
    if transition == ("NOMBRE_PODCAST", "NOMBRE_CAPITULO"):
        return "jingle1.wav"
    if transition == ("NOMBRE_CAPITULO", "INTRO_CAPITULO"):
        return "jingle2.wav"
    if transition in (("INTRO_CAPITULO", "SECCION"), ("SECCION", "SECCION")):
        return "jingle3.wav"
    if transition == ("SECCION", "FINAL"):
        return FINAL_JINGLE_FILENAME
    return None


def required_podcast_assets(blocks: list[dict[str, str | int]]) -> set[str]:
    assets = {
        filename
        for block in blocks
        if (filename := podcast_background_filename(block))
    }
    assets.update(
        filename
        for current_block, next_block in zip(blocks, blocks[1:])
        if (filename := podcast_transition_filename(current_block, next_block))
    )
    if blocks and blocks[-1]["tag"] == "FINAL":
        assets.add(FINAL_JINGLE_FILENAME)
    return assets


def load_audio(path: Path, target_sample_rate: int) -> torch.Tensor:
    audio, sample_rate = ta.load(str(path))
    audio = audio.to(dtype=torch.float32)
    if sample_rate != target_sample_rate:
        audio = ta.functional.resample(audio, sample_rate, target_sample_rate)
    return audio


def as_stereo(audio: torch.Tensor) -> torch.Tensor:
    if audio.shape[0] == 1:
        return audio.repeat(2, 1)
    return audio[:2]


def loop_audio(audio: torch.Tensor, target_length: int) -> torch.Tensor:
    if audio.shape[-1] == 0:
        raise ValueError("Music file contains no audio samples.")
    repeats = (target_length + audio.shape[-1] - 1) // audio.shape[-1]
    return audio.repeat(1, repeats)[..., :target_length]


def mix_voice_with_ducked_music(
    voice: torch.Tensor,
    music: torch.Tensor,
    sample_rate: int,
    fade_out_seconds: float = 1.0,
) -> torch.Tensor:
    """Mix voice with an energy-driven music envelope for radio-style ducking."""
    voice = as_stereo(voice)
    music = as_stereo(music)
    music = loop_audio(music, voice.shape[-1])
    frame_length = max(1, int(sample_rate * 0.04))
    voice_mono = voice.mean(dim=0, keepdim=True)
    voice_energy = torch.sqrt(
        torch.nn.functional.avg_pool1d(
            voice_mono.abs().pow(2),
            kernel_size=frame_length,
            stride=frame_length,
            ceil_mode=True,
        ) + 1e-9
    ).flatten()
    activity_threshold = max(10 ** (-42 / 20), float(voice_energy.max()) * 0.08)
    target_gain = torch.where(
        voice_energy >= activity_threshold,
        torch.tensor(0.10),
        torch.tensor(0.32),
    )

    smoothed_gain = target_gain.clone()
    attack = 0.55
    release = 0.12
    for index in range(1, len(smoothed_gain)):
        coefficient = attack if target_gain[index] < smoothed_gain[index - 1] else release
        smoothed_gain[index] = (
            smoothed_gain[index - 1]
            + coefficient * (target_gain[index] - smoothed_gain[index - 1])
        )

    gain_envelope = torch.nn.functional.interpolate(
        smoothed_gain.view(1, 1, -1),
        size=voice.shape[-1],
        mode="linear",
        align_corners=False,
    ).view(1, -1)
    fade_samples = min(voice.shape[-1], int(sample_rate * fade_out_seconds))
    if fade_samples > 0:
        gain_envelope[..., -fade_samples:] *= torch.linspace(1, 0, fade_samples)

    mixed = voice + music * gain_envelope
    peak = float(mixed.abs().max())
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed


def mix_podcast_blocks(
    blocks: list[dict[str, str | int]],
    block_paths: list[Path],
    output_path: Path,
    sample_rate: int,
    gap_sec: float = 0.0,
) -> list[tuple[int, int]]:
    """Mix the blocks and return each block's (start, end) sample range in the output."""
    block_audio = []
    for block, block_path in zip(blocks, block_paths):
        voice = as_stereo(load_audio(block_path, sample_rate))
        if background_filename := podcast_background_filename(block):
            music = load_audio(PODCAST_MUSIC_DIR / background_filename, sample_rate)
            voice = mix_voice_with_ducked_music(voice, music, sample_rate)
        block_audio.append(voice)

    mixed_parts = [block_audio[0]]
    cursor = block_audio[0].shape[-1]
    block_ranges = [(0, cursor)]
    jingle_overlap_samples = int(sample_rate * 2.0)
    gap_samples = round(validate_gap(gap_sec) * sample_rate)
    for index, next_audio in enumerate(block_audio[1:]):
        if gap_samples:
            mixed_parts.append(torch.zeros(2, gap_samples))
            cursor += gap_samples
        transition_filename = podcast_transition_filename(blocks[index], blocks[index + 1])
        if not transition_filename:
            mixed_parts.append(next_audio)
        else:
            jingle = as_stereo(load_audio(PODCAST_MUSIC_DIR / transition_filename, sample_rate))
            overlap_samples = min(
                jingle_overlap_samples,
                jingle.shape[-1],
                next_audio.shape[-1],
            )
            if overlap_samples == 0:
                mixed_parts.extend((jingle, next_audio))
                cursor += jingle.shape[-1]
            else:
                if jingle.shape[-1] > overlap_samples:
                    mixed_parts.append(jingle[..., :-overlap_samples])
                    cursor += jingle.shape[-1] - overlap_samples
                overlap = jingle[..., -overlap_samples:] + next_audio[..., :overlap_samples]
                peak = float(overlap.abs().max())
                if peak > 0.98:
                    overlap = overlap * (0.98 / peak)
                mixed_parts.extend((overlap, next_audio[..., overlap_samples:]))
        block_ranges.append((cursor, cursor + next_audio.shape[-1]))
        cursor += next_audio.shape[-1]

    if blocks[-1]["tag"] == "FINAL":
        closing_jingle = as_stereo(load_audio(PODCAST_MUSIC_DIR / FINAL_JINGLE_FILENAME, sample_rate))
        mixed_parts.append(closing_jingle)
        block_ranges[-1] = (block_ranges[-1][0], cursor + closing_jingle.shape[-1])

    temporary_output_path = output_path.with_suffix(".tmp.wav")
    ta.save(str(temporary_output_path), torch.cat(mixed_parts, dim=-1), sample_rate)
    temporary_output_path.replace(output_path)
    return block_ranges


def normalize_voice_loudness(
    audio: torch.Tensor,
    sample_rate: int,
    target_dbfs: float = VOICE_TARGET_DBFS,
    peak_limit: float = 0.98,
) -> torch.Tensor:
    """Scale speech so its RMS over non-silent frames matches target_dbfs."""
    frame_length = max(1, int(sample_rate * 0.05))
    frame_rms = torch.sqrt(
        torch.nn.functional.avg_pool1d(
            audio.mean(dim=0, keepdim=True).pow(2),
            kernel_size=frame_length,
            stride=frame_length,
            ceil_mode=True,
        ) + 1e-12
    ).flatten()
    active_frames = frame_rms[frame_rms > 10 ** (-50 / 20)]
    if active_frames.numel() == 0:
        return audio
    speech_rms = float(torch.sqrt(active_frames.pow(2).mean()))
    gain = (10 ** (target_dbfs / 20)) / speech_rms
    peak = float(audio.abs().max()) * gain
    if peak > peak_limit:
        gain *= peak_limit / peak
    return audio * gain


def concatenate_audio_files(
    paths: list[Path],
    output_path: Path,
    sample_rate: int,
    pause_seconds: float = 0.25,
) -> None:
    pause = torch.zeros(1, int(sample_rate * pause_seconds))
    combined_parts = []
    for index, path in enumerate(paths):
        audio = load_audio(path, sample_rate)
        if index:
            combined_parts.append(pause.to(dtype=audio.dtype))
        combined_parts.append(audio)

    combined = normalize_voice_loudness(torch.cat(combined_parts, dim=-1), sample_rate)
    temporary_output_path = output_path.with_suffix(".tmp.wav")
    ta.save(str(temporary_output_path), combined, sample_rate)
    temporary_output_path.replace(output_path)


def convert_wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate_kbps: int = MP3_BITRATE_KBPS) -> None:
    audio, sample_rate = ta.load(str(wav_path))
    channels = min(audio.shape[0], 2)
    pcm = (audio[:channels].clamp(-1, 1) * 32767).to(torch.int16).T.contiguous().numpy()

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate_kbps)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(2)
    mp3_data = encoder.encode(pcm.tobytes()) + encoder.flush()

    temporary_mp3_path = mp3_path.with_suffix(".tmp.mp3")
    temporary_mp3_path.write_bytes(mp3_data)
    temporary_mp3_path.replace(mp3_path)


def export_quiz_audio(manifest: dict, output_dir: Path) -> list[dict[str, str]]:
    outputs = []
    for block in manifest.get("quiz_outputs", []):
        quiz_dir = output_dir / "quiz"
        quiz_dir.mkdir(parents=True, exist_ok=True)
        mp3_path = quiz_dir / f"{block['tag'].lower()}.mp3"
        convert_wav_to_mp3(Path(block["path"]), mp3_path)
        outputs.append({"tag": block["tag"], "path": str(mp3_path.resolve())})
    return outputs


def list_script_files(scripts_dir: str | None) -> list[Path]:
    if not scripts_dir or not scripts_dir.strip():
        return []
    directory = Path(scripts_dir.strip())
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() == ".txt"
            and PROCESSED_SCRIPTS_DIRNAME not in path.relative_to(directory).parts[:-1]
        ),
        key=lambda path: path.name.lower(),
    )


def parse_script_name(path: Path) -> dict[str, str | int] | None:
    match = SCRIPT_NAME_PATTERN.match(path.stem)
    if not match:
        return None
    return {
        "podcast": match["podcast"],
        "language": match["language"].lower(),
        "chapter": int(match["chapter"]),
        "block": int(match["block"] or 1),
        "slug": match["slug"],
    }


def humanize_slug(slug: str) -> str:
    text = " ".join(slug.replace("-", " ").replace("_", " ").split())
    return text[:1].upper() + text[1:]


def clean_title(text: str) -> str:
    text = text.strip().rstrip(".").strip()
    if text.isupper():
        text = ". ".join(sentence[:1].upper() + sentence[1:] for sentence in text.lower().split(". "))
    return text


def script_block_title(
    tag_blocks: list[dict[str, str | int]],
    fallback: str,
    tags: tuple[str, ...] = ("SECCION", "FINAL", "NOMBRE_CAPITULO"),
) -> str:
    for tag in tags:
        for block in tag_blocks:
            if block["tag"] == tag and str(block["text"]).strip():
                return clean_title(str(block["text"]).strip().splitlines()[0])
    return fallback


def t3_model_filename(language_id: str) -> str:
    model_key = LANGUAGE_MODELS[language_id]["key"]
    if model_key == "en":
        return "t3_cfg.safetensors"
    return MULTILINGUAL_T3_MODELS.get(model_key, model_key)


def group_chapter_scripts(script_files: list[Path]) -> dict[tuple[str, str, int], list[tuple[dict, Path]]]:
    chapters: dict[tuple[str, str, int], list[tuple[dict, Path]]] = {}
    for script_path in script_files:
        script_info = parse_script_name(script_path)
        if script_info is None:
            print(f"Ignoring {script_path.name}: not a chapter block script")
            continue
        key = (script_info["podcast"], script_info["language"], script_info["chapter"])
        chapters.setdefault(key, []).append((script_info, script_path))
    for blocks in chapters.values():
        blocks.sort(key=lambda item: item[0]["block"])
    return dict(sorted(chapters.items()))


def move_to_processed(script_path: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    script_path.replace(processed_dir / script_path.name)


def read_script_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252")


def describe_script_files(scripts_dir: str | None):
    if not scripts_dir or not scripts_dir.strip():
        return gr.update(choices=[], value=[], info=None)
    directory = Path(scripts_dir.strip())
    if not directory.is_dir():
        return gr.update(choices=[], value=[], info="El directorio no existe.")
    script_files = list_script_files(scripts_dir)
    valid_scripts = [path for path in script_files if parse_script_name(path)]
    invalid_scripts = [path for path in script_files if not parse_script_name(path)]
    if not valid_scripts:
        info = "No hay guiones .txt pendientes en este directorio."
    else:
        info = f"{len(valid_scripts)} guion(es). Desmarca los que no quieras generar todavía."
    if invalid_scripts:
        names = ", ".join(str(path.relative_to(directory)) for path in invalid_scripts)
        info += f" Se ignoran por nombre no válido: {names}"
    choices = [str(path.relative_to(directory)) for path in valid_scripts]
    return gr.update(choices=choices, value=choices, info=info)


def browse_scripts_directory(current_dir: str | None) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=current_dir or None,
            title="Selecciona el directorio de guiones (.txt)",
        )
        root.destroy()
    except Exception as e:
        print(f"Could not open directory dialog: {e}")
        return current_dir or ""
    return selected or current_dir or ""


def synthesize_script(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
    speed_input: float = 1.0,
    progress=gr.Progress(),
    progress_prefix: str = "",
    remember_request: bool = True,
    gap_sec: float = 0.0,
) -> Path:
    """Synthesize one script into a resumable job and return the combined WAV path."""
    try:
        gap_sec = validate_gap(gap_sec)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    current_model = get_or_load_model(language_id)

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    first_podcast_tag = PODCAST_BLOCK_PATTERN.search(text_input)
    if first_podcast_tag and text_input[:first_podcast_tag.start()].strip():
        raise gr.Error("Place the first podcast tag before any text to avoid losing content.")

    podcast_blocks = parse_podcast_blocks(text_input)
    theory_blocks, quiz_blocks = split_podcast_blocks(podcast_blocks)
    empty_blocks = [str(block["tag"]) for block in podcast_blocks if not block["text"]]
    if empty_blocks:
        raise gr.Error(f"Podcast blocks without text: {', '.join(empty_blocks)}")
    generation_blocks = podcast_blocks or [{
        "tag": "TEXT",
        "occurrence": 1,
        "text": text_input,
    }]
    block_chunks = [split_text_into_chunks(str(block["text"])) for block in generation_blocks]
    work_items = [
        {
            "block_index": block_index,
            "chunk_index": chunk_index,
            "text": chunk,
        }
        for block_index, chunks in enumerate(block_chunks, start=1)
        for chunk_index, chunk in enumerate(chunks, start=1)
    ]
    if not work_items:
        raise gr.Error("Enter some text to synthesize.")

    required_assets = required_podcast_assets(theory_blocks)
    missing_assets = sorted(
        filename
        for filename in required_assets
        if not (PODCAST_MUSIC_DIR / filename).is_file()
    )
    if missing_assets:
        missing_list = ", ".join(missing_assets)
        raise gr.Error(
            f"Missing podcast audio files in {PODCAST_MUSIC_DIR}: {missing_list}"
        )

    music_assets = {
        filename: sha256_file(PODCAST_MUSIC_DIR / filename)
        for filename in sorted(required_assets)
    }

    chosen_prompt = persist_reference_audio(
        audio_prompt_path_input or default_audio_for_ui(language_id)
    )
    reference_key = chosen_prompt or "default"
    if chosen_prompt and Path(chosen_prompt).is_file():
        reference_key = hashlib.sha256(Path(chosen_prompt).read_bytes()).hexdigest()

    request_data = {
        "text": text_input,
        "language_id": language_id,
        "reference": chosen_prompt,
        "reference_key": reference_key,
        "exaggeration": float(exaggeration_input),
        "temperature": float(temperature_input),
        "seed": int(seed_num_input),
        "cfg_weight": float(cfgw_input),
        "t3_model": LANGUAGE_MODELS[language_id]["key"],
        "chunks": [item["text"] for item in work_items],
        "podcast_blocks": podcast_blocks,
        "podcast_mix_version": PODCAST_MIX_VERSION if podcast_blocks else 1,
        "music_assets": music_assets,
    }
    # Only included when changed so existing jobs keep their hash and cached chunks.
    if float(speed_input) != 1.0:
        request_data["speed"] = float(speed_input)
    if gap_sec:
        request_data["gap_sec"] = gap_sec
    job_payload = json.dumps(request_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    job_id = hashlib.sha256(job_payload).hexdigest()[:16]
    job_dir = CACHE_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    combined_path = job_dir / "combined.wav"
    manifest_path = job_dir / "manifest.json"
    if remember_request:
        save_json(LAST_REQUEST_PATH, request_data)

    previous_manifest = load_json(manifest_path) or {}
    if (
        combined_path.is_file()
        and (not podcast_blocks or "segments_sec" in previous_manifest)
        and previous_manifest.get("voice_target_dbfs") == VOICE_TARGET_DBFS
        and len(previous_manifest.get("quiz_outputs", [])) == len(quiz_blocks)
        and all(Path(output["path"]).is_file() for output in previous_manifest.get("quiz_outputs", []))
    ):
        progress(1, desc=f"{progress_prefix}Recovered completed audio")
        print(f"Recovered completed job {job_id}: {combined_path}")
        return combined_path.resolve()

    manifest = {
        **request_data,
        "job_id": job_id,
        "status": "running",
        "completed_chunks": 0,
        "total_chunks": len(work_items),
        "output": str(combined_path.resolve()),
    }
    save_json(manifest_path, manifest)
    print(f"Job {job_id}: {len(work_items)} chunks. Files: {job_dir}")

    generate_kwargs = {
        "exaggeration": exaggeration_input,
        "temperature": temperature_input,
        "cfg_weight": cfgw_input,
        "speed": float(speed_input),
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print("Using configured reference audio.")
    else:
        print("No audio prompt provided; using default voice.")
        
    chunk_paths = []
    started_at = time.monotonic()
    generated_this_run = 0
    for index, work_item in enumerate(work_items, start=1):
        block_index = int(work_item["block_index"])
        chunk_index = int(work_item["chunk_index"])
        chunk = str(work_item["text"])
        if podcast_blocks:
            block = generation_blocks[block_index - 1]
            block_stem = Path(podcast_block_filename(block, block_index)).stem
            chunk_path = job_dir / f"{block_stem}-chunk-{chunk_index:04d}.wav"
        else:
            chunk_path = job_dir / f"chunk-{index:04d}.wav"
        chunk_paths.append(chunk_path)
        if chunk_path.is_file():
            try:
                _, sample_rate = ta.load(str(chunk_path))
                if sample_rate == current_model.sr:
                    manifest["completed_chunks"] = index
                    save_json(manifest_path, manifest)
                    print(f"Reusing saved part {index}/{len(work_items)}")
                    progress((index, len(work_items)), desc=f"{progress_prefix}Recovered part {index}/{len(work_items)}")
                    continue
            except Exception:
                chunk_path.unlink(missing_ok=True)

        elapsed = time.monotonic() - started_at
        remaining_seconds = (elapsed / generated_this_run) * (len(work_items) - index + 1) if generated_this_run > 0 else None
        eta = f" ETA: {remaining_seconds / 60:.1f} min" if remaining_seconds is not None else ""
        progress((index - 1, len(work_items)), desc=f"{progress_prefix}Generating part {index}/{len(work_items)}.{eta}")
        print(f"Generating part {index}/{len(work_items)}: '{chunk[:50]}...'")
        if language_id == "en":
            wav = current_model.generate(chunk, **generate_kwargs)
        else:
            wav = current_model.generate(
                chunk,
                language_id=language_id,
                **generate_kwargs
            )
        temporary_chunk_path = chunk_path.with_suffix(".tmp.wav")
        ta.save(str(temporary_chunk_path), wav.detach().cpu(), current_model.sr)
        temporary_chunk_path.replace(chunk_path)
        generated_this_run += 1
        manifest["completed_chunks"] = index
        save_json(manifest_path, manifest)

    if podcast_blocks:
        block_paths = []
        for block_index, (block, chunks) in enumerate(
            zip(generation_blocks, block_chunks),
            start=1,
        ):
            block_path = job_dir / podcast_block_filename(block, block_index)
            first_chunk_index = sum(len(previous_chunks) for previous_chunks in block_chunks[:block_index - 1])
            block_chunk_paths = chunk_paths[first_chunk_index:first_chunk_index + len(chunks)]
            concatenate_audio_files(block_chunk_paths, block_path, current_model.sr)
            block_paths.append(block_path)
        block_ranges = mix_podcast_blocks(
            theory_blocks, block_paths[:len(theory_blocks)], combined_path, current_model.sr,
            gap_sec=gap_sec,
        )
        manifest["segments_sec"] = [
            [start / current_model.sr, end / current_model.sr]
            for start, end in block_ranges
        ]
        manifest["block_outputs"] = [str(path.resolve()) for path in block_paths]
        manifest["quiz_outputs"] = [
            {"tag": str(block["tag"]), "path": str(path.resolve())}
            for block, path in zip(quiz_blocks, block_paths[len(theory_blocks):])
        ]
    else:
        concatenate_audio_files(chunk_paths, combined_path, current_model.sr)
    manifest["voice_target_dbfs"] = VOICE_TARGET_DBFS
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)

    progress(1, desc=f"{progress_prefix}Audio complete")
    print(f"All audio parts generated and combined: {combined_path}")
    return combined_path.resolve()


@contextmanager
def keep_awake():
    """Keep Windows from sleeping or turning off the display while active."""
    if sys.platform != "win32":
        yield
        return
    import ctypes

    es_continuous = 0x80000000
    es_system_required = 0x00000001
    es_display_required = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        es_continuous | es_system_required | es_display_required
    )
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


@keep_awake()
def generate_tts_audio(
    text_input: str,
    scripts_dir: str,
    selected_scripts: list[str] | None,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
    speed_input: float = 1.0,
    version_input: int = 1,
    gap_sec_input: float = 0.0,
    progress=gr.Progress(),
):
    """
    Generate speech from text, or from every .txt script in a directory when the text is empty.

    With text, a combined WAV is produced. With a scripts directory, the block scripts
    {podcast}__{idioma}__c{NN}-b{NN}__{titulo}.txt of each chapter are joined in block order and
    synthesized in their own language into publish/{podcast}/{idioma}/ch{NN}/v{N}/ch{NN}.mp3 plus a
    ch{NN}.json with block timings and voice-only quiz/*.mp3; the used scripts are then
    moved to processed/{podcast}/{idioma}/ch{NN}/v{N}/.
    Intermediate WAV files are deleted once the MP3 is written. Interrupted jobs keep their parts.

    Args:
        text_input (str): The text to synthesize into speech (maximum 100,000 characters). Leave empty to use scripts_dir.
        scripts_dir (str): Directory containing .txt scripts, used only when text_input is empty.
        selected_scripts (list[str] | None): Script paths relative to scripts_dir to generate. None generates all of them.
        language_id (str): The language code for synthesis (es, en, fr)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer.
        speed_input (float, optional): Speech tempo multiplier (0.75-1.25, <1 slower). Pitch is preserved. Defaults to 1.0.
        version_input (int): Positive publication version. Existing versions are immutable.
        gap_sec_input (float): Additional silence between theory blocks, excluding variable jingles.

    Returns:
        tuple: Path to the combined WAV (text mode) and newline-separated generated MP3 paths (directory mode).
    """
    try:
        version = positive_version(version_input)
        gap_sec = validate_gap(gap_sec_input)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    generation_args = (
        language_id,
        audio_prompt_path_input,
        exaggeration_input,
        temperature_input,
        seed_num_input,
        cfgw_input,
        speed_input,
    )
    if text_input and text_input.strip():
        combined_path = synthesize_script(text_input, *generation_args, progress=progress, gap_sec=gap_sec)
        manifest = load_json(combined_path.parent / "manifest.json")
        if manifest is None:
            raise gr.Error(f"No se puede leer el manifiesto de audio: {combined_path.parent}")
        quiz_outputs = export_quiz_audio(manifest, combined_path.parent)
        return str(combined_path), "\n".join(output["path"] for output in quiz_outputs)

    script_files = list_script_files(scripts_dir)
    if not script_files:
        raise gr.Error("No hay texto ni un directorio con ficheros .txt disponibles.")

    chapters = group_chapter_scripts(script_files)
    if not chapters:
        raise gr.Error("No hay guiones con formato {podcast}__{idioma}__c{NN}[-b{NN}]__{titulo}.txt en el directorio.")

    if selected_scripts is not None:
        directory = Path(scripts_dir.strip())
        selected = set(selected_scripts)
        selected_chapters = {}
        for key, chapter_scripts in chapters.items():
            selected_count = sum(
                str(script_path.relative_to(directory)) in selected for _, script_path in chapter_scripts
            )
            if selected_count == len(chapter_scripts):
                selected_chapters[key] = chapter_scripts
            elif selected_count:
                podcast, chapter_language, chapter = key
                gr.Warning(
                    f"{podcast}/{chapter_language}/ch{chapter:02d}: solo hay marcados algunos bloques del capítulo; se omite."
                )
        chapters = selected_chapters
        if not chapters:
            raise gr.Error("No hay ningún capítulo completo seleccionado.")

    mp3_paths = []
    for chapter_index, ((podcast, chapter_language, chapter), chapter_scripts) in enumerate(chapters.items(), start=1):
        chapter_name = f"{podcast}/{chapter_language}/ch{chapter:02d}"
        prefix = f"[{chapter_index}/{len(chapters)}] {chapter_name}: "
        script_paths = [script_path for _, script_path in chapter_scripts]
        if chapter_language not in LANGUAGE_MODELS:
            gr.Warning(f"{chapter_name}: idioma '{chapter_language}' no soportado; se omite.")
            continue
        block_numbers = [info["block"] for info, _ in chapter_scripts]
        if len(set(block_numbers)) != len(block_numbers):
            gr.Warning(f"{chapter_name}: hay números de bloque repetidos; se omite.")
            continue

        try:
            chapter_prefix(podcast, chapter_language, chapter, version)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        chapter_relative_dir = Path(podcast, chapter_language, f"ch{chapter:02d}")
        output_dir = PUBLISH_DIR / chapter_relative_dir / f"v{version}"
        processed_dir = Path(scripts_dir.strip()) / PROCESSED_SCRIPTS_DIRNAME / chapter_relative_dir / f"v{version}"
        mp3_path = output_dir / f"ch{chapter:02d}.mp3"
        json_path = output_dir / f"ch{chapter:02d}.json"
        script_texts = [read_script_file(script_path) for script_path in script_paths]
        script_tag_blocks = []
        invalid_scripts = []
        for script_path, script_text in zip(script_paths, script_texts):
            first_tag = PODCAST_BLOCK_PATTERN.search(script_text)
            if not first_tag or script_text[:first_tag.start()].strip():
                invalid_scripts.append(script_path.name)
            script_tag_blocks.append([
                block for block in parse_podcast_blocks(script_text)
                if block["tag"] in PODCAST_BLOCK_TAGS
            ])
        if invalid_scripts:
            gr.Warning(f"{chapter_name}: estos guiones deben empezar por una etiqueta de podcast: {', '.join(invalid_scripts)}; se omite.")
            continue

        theory_blocks, quiz_blocks = split_podcast_blocks(parse_podcast_blocks("\n\n".join(script_texts)))
        chapter_title = script_block_title(
            theory_blocks, "", tags=("NOMBRE_CAPITULO",),
        )
        if not chapter_title:
            raise gr.Error(f"{chapter_name}: falta el titulo en [NOMBRE_CAPITULO].")
        try:
            quiz_entries = quiz_metadata(quiz_blocks)
            chosen_reference = persist_reference_audio(
                audio_prompt_path_input or default_audio_for_ui(chapter_language)
            )
            voice = reference_voice_id(chosen_reference)
            model_filename = t3_model_filename(chapter_language)
            if not model_filename:
                raise ValueError("Falta el modelo TTS configurado.")
            missing_assets = [
                filename for filename in sorted(required_podcast_assets(theory_blocks))
                if not (PODCAST_MUSIC_DIR / filename).is_file()
            ]
            if missing_assets:
                raise ValueError(f"Faltan archivos de musica: {', '.join(missing_assets)}")
            generation_config = {
                "tts": "chatterbox.tts.ChatterboxTTS" if chapter_language == "en" else "chatterbox.mtl_tts.ChatterboxMultilingualTTS",
                "model": model_filename,
                "voice": voice,
                "gap_sec": gap_sec,
                "scripts_hash": scripts_hash(script_paths),
                "podcast_mix_version": PODCAST_MIX_VERSION,
            }
            fingerprint = request_hash({
                **generation_config,
                "schema_version": 1,
                "exaggeration": float(exaggeration_input),
                "temperature": float(temperature_input),
                "seed": int(seed_num_input),
                "cfg_weight": float(cfgw_input),
                "speed": float(speed_input),
                "bitrate_kbps": MP3_BITRATE_KBPS,
                "music": {
                    filename: sha256_file(PODCAST_MUSIC_DIR / filename)
                    for filename in sorted(required_podcast_assets(theory_blocks))
                },
            })
        except ValueError as error:
            raise gr.Error(f"{chapter_name}: {error}") from error
        generation_config["request_hash"] = fingerprint
        quiz_paths = [output_dir / "quiz" / f"{str(block['tag']).lower()}.mp3" for block in quiz_blocks]
        if output_dir.exists():
            published = load_json(json_path)
            try:
                if published is None:
                    raise ValueError("La version ya existe con otra configuracion o sin un manifest verificable.")
                validate_chapter_manifest(published)
                if published["generated"].get("request_hash") != fingerprint:
                    raise ValueError("La version ya existe con otra configuracion.")
                if (
                    not mp3_path.is_file()
                    or mp3_metadata(mp3_path) != {key: published["audio"][key] for key in ("duration_sec", "bitrate_kbps", "sha256")}
                    or not all(path.is_file() for path in quiz_paths)
                ):
                    raise ValueError("La version existente esta incompleta o el MP3 ha cambiado.")
            except ValueError as error:
                raise gr.Error(f"{chapter_name}/v{version}: {error} Selecciona una version nueva.") from error
            print(f"Reusing verified publication {chapter_name}/v{version}")
            for script_path in script_paths:
                move_to_processed(script_path, processed_dir)
            mp3_paths.extend(str(path.resolve()) for path in [mp3_path, *quiz_paths])
            continue

        print(f"Processing chapter {chapter_name}: {', '.join(path.name for path in script_paths)}")
        combined_path = synthesize_script(
            "\n\n".join(script_texts),
            chapter_language,
            chosen_reference,
            *generation_args[2:],
            progress=progress,
            progress_prefix=prefix,
            remember_request=False,
            gap_sec=gap_sec,
        )
        manifest = load_json(combined_path.parent / "manifest.json")
        if manifest is None:
            raise gr.Error(f"No se puede leer el manifiesto de audio: {combined_path.parent}")
        segments = manifest["segments_sec"]

        blocks_metadata = []
        segment_index = 0
        if len(segments) != sum(len(blocks) for blocks in script_tag_blocks):
            raise gr.Error(f"{chapter_name}: los tiempos no corresponden a los bloques de teoria.")
        for (_, script_path), tag_blocks in zip(chapter_scripts, script_tag_blocks):
            file_segments = segments[segment_index:segment_index + len(tag_blocks)]
            segment_index += len(tag_blocks)
            for tag_block, segment in zip(tag_blocks, file_segments):
                blocks_metadata.append({
                    "order": len(blocks_metadata) + 1,
                    "title": clean_title(str(tag_block["text"]).splitlines()[0]),
                    "script": script_path.name,
                    "start_sec": segment[0],
                    "end_sec": segment[1],
                })

        progress(1, desc=f"{prefix}Converting to MP3")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f".v{version}-", dir=output_dir.parent))
        try:
            staged_mp3 = staging_dir / mp3_path.name
            convert_wav_to_mp3(combined_path, staged_mp3)
            quiz_outputs = export_quiz_audio(manifest, staging_dir)
            if (
                {output["tag"] for output in quiz_outputs} != {entry["tag"] for entry in quiz_entries}
                or not all(Path(output["path"]).is_file() for output in quiz_outputs)
            ):
                raise gr.Error(f"{chapter_name}: faltan segmentos de audio del quiz.")
            try:
                published = build_chapter_manifest(
                    podcast=podcast, language=chapter_language, chapter=chapter,
                    version=version, title=chapter_title, mp3_path=staged_mp3,
                    blocks=blocks_metadata, quiz=quiz_entries, generated=generation_config,
                )
            except ValueError as error:
                raise gr.Error(f"{chapter_name}: {error}") from error
            save_json(staging_dir / json_path.name, published)
            staging_dir.rename(output_dir)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

        shutil.rmtree(combined_path.parent, ignore_errors=True)
        for script_path in script_paths:
            move_to_processed(script_path, processed_dir)
        print(f"Saved {mp3_path} and {json_path.name}; moved scripts to {processed_dir}")
        mp3_paths.append(str(mp3_path.resolve()))
        mp3_paths.extend(str(path.resolve()) for path in quiz_paths)

    progress(1, desc="All scripts complete")
    return gr.update(), "\n".join(mp3_paths)

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox TTS
        Síntesis de voz con modelos optimizados para español, inglés y francés.
        """
    )

    with gr.Row():
        with gr.Column():
            initial_lang = "es"
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (max chars 100,000)",
                max_lines=12,
                max_length=100000
            )
            with gr.Row():
                scripts_dir = gr.Textbox(
                    label="Directorio de guiones (.txt)",
                    info="Se usa solo si el cuadro de texto está vacío. Formato: {podcast}__{idioma}__c{NN}[-b{NN}]__{titulo}.txt. Salida: publish/{podcast}/{idioma}/ch{NN}/v{N}/ con MP3, manifest y quiz/. Las versiones publicadas no se sobrescriben.",
                    placeholder="C:\\ruta\\a\\guiones",
                    scale=4,
                )
                browse_btn = gr.Button("Examinar...", scale=1)
            scripts_list = gr.CheckboxGroup(
                label="Guiones encontrados",
                choices=[],
            )
            browse_btn.click(
                fn=browse_scripts_directory,
                inputs=[scripts_dir],
                outputs=[scripts_dir],
                show_progress=False,
            )
            scripts_dir.change(
                fn=describe_script_files,
                inputs=[scripts_dir],
                outputs=[scripts_list],
                show_progress=False,
            )
            with gr.Row():
                publication_version = gr.Number(
                    label="Versión de publicación", value=1, minimum=1, step=1, precision=0,
                )
                block_gap = gr.Number(
                    label="Silencio entre bloques (segundos)", value=0, minimum=0,
                )
            gr.Markdown("El silencio se añade solo entre bloques de teoría; los jingles tienen duración variable.")
            
            language_id = gr.Dropdown(
                choices=[(name, code) for code, name in UI_LANGUAGES.items()],
                value=initial_lang,
                label="Idioma",
                info="Selecciona el idioma para la síntesis de voz"
            )
            model_status = gr.Markdown(
                f"**Modelo activo:** {LANGUAGE_MODELS[initial_lang]['name']}"
            )
            
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Audio de referencia (obligatorio para publicar capítulos)",
                value=default_audio_for_ui(initial_lang)
            )
            ref_wav.upload(
                fn=persist_reference_audio,
                inputs=[ref_wav],
                outputs=[ref_wav],
                show_progress=False,
            )
            
            gr.Markdown(
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                elem_classes=["audio-note"]
            )
            
            exaggeration = gr.Slider(
                0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=last_request_value("exaggeration", .5)
            )
            cfg_weight = gr.Slider(
                0.2, 1, step=.05, label="CFG/Pace", value=last_request_value("cfg_weight", .5)
            )
            speed = gr.Slider(
                0.75, 1.25, step=.05, label="Velocidad (1 = normal, <1 más lento)", value=last_request_value("speed", 1.0)
            )

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=last_request_value("seed", 0), label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="Temperature", value=last_request_value("temperature", .8))

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(
                value=get_latest_combined_path(),
                label="Combined output audio",
                format="wav",
            )
            mp3_output = gr.Textbox(
                label="MP3 generados (capítulos y quiz)",
                lines=4,
                interactive=False,
            )
            gr.Markdown(f"Fragmentos temporales en `{CACHE_DIR}`; capítulos listos para publicar en `{PUBLISH_DIR}`")
            gr.Markdown(f"Los jingles y fondos del podcast se leen desde `{PODCAST_MUSIC_DIR}`")

        def on_language_change(lang, current_ref, current_text):
            get_or_load_model(lang)
            return (
                default_audio_for_ui(lang),
                default_text_for_ui(lang),
                f"**Modelo activo:** {LANGUAGE_MODELS[lang]['name']}",
            )

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text, model_status],
            show_progress="full"
        )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            scripts_dir,
            scripts_list,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
            speed,
            publication_version,
            block_gap,
        ],
        outputs=[audio_output, mp3_output],
    ).then(
        fn=describe_script_files,
        inputs=[scripts_dir],
        outputs=[scripts_list],
        show_progress=False,
    )

demo.launch(mcp_server=True)
