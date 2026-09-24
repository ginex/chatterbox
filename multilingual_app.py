import random
import os
import re
import time
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T3_MODEL = os.getenv("CHATTERBOX_MULTILINGUAL_T3_MODEL", "es-es")
print(f"🚀 Running on device: {DEVICE}")
print(f"Using multilingual T3 model: {T3_MODEL}")

# --- Global Model Initialization ---
MODEL = None
APP_DATA_DIR = Path(__file__).resolve().parent / "chatterbox_output"
JOBS_DIR = APP_DATA_DIR / "jobs"
REFERENCE_DIR = APP_DATA_DIR / "reference"
PODCAST_MUSIC_DIR = Path(__file__).resolve().parent / "podcast_music"
LAST_REQUEST_PATH = APP_DATA_DIR / "last_request.json"
APP_DATA_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
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
    rf"\[({'|'.join(PODCAST_BLOCK_TAGS)})\]",
    flags=re.IGNORECASE,
)

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
    combined_files = list(JOBS_DIR.glob("*/combined.wav"))
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


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")
    
    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])
    
    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(DEVICE, t3_model=T3_MODEL)
            if hasattr(MODEL, 'to') and str(MODEL.device) != DEVICE:
                MODEL.to(DEVICE)
            print(f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL

# Attempt to load the model at startup.
try:
    get_or_load_model()
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


def split_text_into_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Split text near max_chars, preferring sentence and clause boundaries."""
    normalized_text = " ".join(text.split())
    if not normalized_text:
        return []

    sentences = re.split(r"(?<=[.!?。！？])\s+", normalized_text)
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

        clauses = re.split(r"(?<=[,;:，；：])\s*", sentence)
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
        return "jingle2.wav"
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
) -> None:
    mixed_parts = []
    for index, (block, block_path) in enumerate(zip(blocks, block_paths)):
        voice = as_stereo(load_audio(block_path, sample_rate))
        if background_filename := podcast_background_filename(block):
            music = load_audio(PODCAST_MUSIC_DIR / background_filename, sample_rate)
            voice = mix_voice_with_ducked_music(voice, music, sample_rate)
        mixed_parts.append(voice)

        if index + 1 < len(blocks):
            transition_filename = podcast_transition_filename(block, blocks[index + 1])
            if transition_filename:
                mixed_parts.append(
                    as_stereo(load_audio(PODCAST_MUSIC_DIR / transition_filename, sample_rate))
                )

    temporary_output_path = output_path.with_suffix(".tmp.wav")
    ta.save(str(temporary_output_path), torch.cat(mixed_parts, dim=-1), sample_rate)
    temporary_output_path.replace(output_path)


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

    temporary_output_path = output_path.with_suffix(".tmp.wav")
    ta.save(str(temporary_output_path), torch.cat(combined_parts, dim=-1), sample_rate)
    temporary_output_path.replace(output_path)


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
    progress=gr.Progress()
) -> str:
    """
    Generate high-quality speech audio from text using Chatterbox Multilingual model with optional reference audio styling.
    Supported languages: English, French, German, Spanish, Italian, Portuguese, and Hindi.
    
    This tool synthesizes natural-sounding speech from input text. When a reference audio file 
    is provided, it captures the speaker's voice characteristics and speaking style. The generated audio 
    maintains the prosody, tone, and vocal qualities of the reference speaker, or uses default voice if no reference is provided.

    Args:
        text_input (str): The text to synthesize into speech (maximum 100,000 characters)
        language_id (str): The language code for synthesis (eg. en, fr, de, es, it, pt, hi)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer. 

    Returns:
        str: Path to the persistent combined WAV file.
    """
    current_model = get_or_load_model()

    if current_model is None:
        raise RuntimeError("TTS model is not loaded.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    text_input = text_input[:100000]
    first_podcast_tag = PODCAST_BLOCK_PATTERN.search(text_input)
    if first_podcast_tag and text_input[:first_podcast_tag.start()].strip():
        raise gr.Error("Place the first podcast tag before any text to avoid losing content.")

    podcast_blocks = parse_podcast_blocks(text_input)
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

    required_assets = required_podcast_assets(podcast_blocks)
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
        filename: {
            "size": (PODCAST_MUSIC_DIR / filename).stat().st_size,
            "modified_ns": (PODCAST_MUSIC_DIR / filename).stat().st_mtime_ns,
        }
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
        "t3_model": T3_MODEL,
        "chunks": [item["text"] for item in work_items],
        "podcast_blocks": podcast_blocks,
        "podcast_mix_version": 2 if podcast_blocks else 1,
        "music_assets": music_assets,
    }
    job_payload = json.dumps(request_data, ensure_ascii=False, sort_keys=True).encode("utf-8")
    job_id = hashlib.sha256(job_payload).hexdigest()[:16]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    combined_path = job_dir / "combined.wav"
    manifest_path = job_dir / "manifest.json"
    save_json(LAST_REQUEST_PATH, request_data)

    if combined_path.is_file():
        progress(1, desc="Recovered completed audio")
        print(f"Recovered completed job {job_id}: {combined_path}")
        return str(combined_path.resolve())

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
    }
    if chosen_prompt:
        generate_kwargs["audio_prompt_path"] = chosen_prompt
        print(f"Using audio prompt: {chosen_prompt}")
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
                    progress((index, len(work_items)), desc=f"Recovered part {index}/{len(work_items)}")
                    continue
            except Exception:
                chunk_path.unlink(missing_ok=True)

        elapsed = time.monotonic() - started_at
        remaining_seconds = (elapsed / generated_this_run) * (len(work_items) - index + 1) if generated_this_run > 0 else None
        eta = f" ETA: {remaining_seconds / 60:.1f} min" if remaining_seconds is not None else ""
        progress((index - 1, len(work_items)), desc=f"Generating part {index}/{len(work_items)}.{eta}")
        print(f"Generating part {index}/{len(work_items)}: '{chunk[:50]}...'")
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
        mix_podcast_blocks(podcast_blocks, block_paths, combined_path, current_model.sr)
        manifest["block_outputs"] = [str(path.resolve()) for path in block_paths]
    else:
        concatenate_audio_files(chunk_paths, combined_path, current_model.sr)
    manifest["status"] = "complete"
    save_json(manifest_path, manifest)

    progress(1, desc="Audio complete")
    print(f"All audio parts generated and combined: {combined_path}")
    return str(combined_path.resolve())

with gr.Blocks() as demo:
    if T3_MODEL == "es-es":
        gr.Markdown(
            """
            # Chatterbox Español (España)
            Síntesis de voz optimizada para español de España con clonación de voz mediante audio de referencia.
            """
        )
        gr.Markdown("**Idioma:** Español de España (`es`)")
    else:
        gr.Markdown(
            """
            # Chatterbox Multilingual Demo
            Generate high-quality multilingual speech from text with reference audio styling, supporting 23 languages.
            """
        )
        gr.Markdown(get_supported_languages_display())

    with gr.Row():
        with gr.Column():
            initial_lang = "es"
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (max chars 100,000)",
                max_lines=12,
                max_length=100000
            )
            
            language_id = gr.Dropdown(
                choices=["es"] if T3_MODEL == "es-es" else list(ChatterboxMultilingualTTS.get_supported_languages().keys()),
                value=initial_lang,
                label="Language",
                info="Select the language for text-to-speech synthesis"
            )
            
            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
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
            gr.Markdown(f"Los trabajos y fragmentos se guardan en `{JOBS_DIR}`")
            gr.Markdown(f"Los jingles y fondos del podcast se leen desde `{PODCAST_MUSIC_DIR}`")

        def on_language_change(lang, current_ref, current_text):
            return default_audio_for_ui(lang), default_text_for_ui(lang)

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text],
            show_progress=False
        )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

demo.launch(mcp_server=True)
