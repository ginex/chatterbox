# Podcast music

Place these WAV files in this directory:

- `jingle1.wav`
- `jingle2.wav`
- `jingle3.wav`
- `background_music_1.wav`
- `background_music_2.wav`
- `background_music_3.wav`
- `background_music_4.wav`
- `final_music.wav`

Files may use any sample rate and may be mono or stereo. The mixer converts them
to the model sample rate, preserves stereo files, and duplicates mono files across
both channels. The generated mono voice is centered in the stereo podcast mix.

The final 2 seconds of each jingle overlap the beginning of the following spoken
block. Background and final music do not overlap when changing between blocks.
After the text of `[FINAL]`, `jingle2.wav` plays again in full, without overlap,
to close the theory audio. Its duration is included in the final block's timing.

## Quiz audio

Scripts may continue after `[FINAL]` with `[PREGUNTAS]`, numbered
`[PREGUNTA_N]`, `[OPCION_CORRECTA_N]`, `[RESPUESTA_N]`, and the shared
`[INCORRECTO]`, `[CORRECTO]`, `[ESPERA_1]`, `[ESPERA_2]`, `[FIN_CAPITULO]`
blocks. Question numbers are positive integers and are not limited to two
questions. Each quiz tag must be unique within a chapter and contain text.
`[OPCION_CORRECTA_N]` contains the answer letter, which is also synthesized.

Directory generation keeps `publish/{podcast}/{idioma}/ch{NN}/v{N}/ch{NN}.mp3`
for theory only, from `[NOMBRE_PODCAST]` through `[FINAL]` and the closing jingle.
Each quiz block is exported next to it in `quiz/`, using the lowercase tag as
the filename, for example `quiz/pregunta_1.mp3`, `quiz/opcion_correcta_1.mp3`,
and `quiz/fin_capitulo.mp3`. These files contain only normalized speech:
no background music, jingles, or crossfades. The chapter JSON lists their tags
and R2 keys under `quiz`; theory block timings exclude quiz audio.

Questions must include the spoken alternatives, each with a unique single-letter
label: `A.`, `A)`, `(A)` or `A:` followed by whitespace and the alternative text.
Labels may be inline or on separate lines. The manifest's `option_letters` lists
those actual labels in source order; it never assumes four alternatives.
`OPCION_CORRECTA_N` must contain exactly one of those letters (case-insensitive);
it becomes uppercase `correct_option` in the manifest. Each question must have
its matching correct-option and explanation blocks. When a quiz is present,
all five shared segments and `FIN_CAPITULO` are required (the shared segments are
`PREGUNTAS`, `INCORRECTO`, `CORRECTO`, `ESPERA_1`, `ESPERA_2`).

The correct-option MP3 is still generated for compatibility, but is not part of
the app's playback sequence. Write explanations that identify and explain the
correct answer without assuming success or failure. Feedback and waiting phrases
must work with any question. The generator does not rewrite these texts, implement
answer handling, or insert response-waiting silences.

In text-input mode the theory output remains a WAV, with quiz MP3s under the
job's `quiz/` directory. Generated MP3 paths are shown in the UI in both modes.

## Chapter manifests (schema version 1)

In Gradio, select **Version de publicacion** (a positive integer, default `1`)
and **Silencio entre bloques (segundos)** (finite, nonnegative, default `0`).
All chapter MP3s and `ch{NN}.json` go under
`publish/{podcast}/{language}/ch{NN}/v{N}/`. Only directory mode publishes a
chapter manifest; text mode retains its existing WAV/quiz workflow.

Publication requires a local, nonempty reference audio file. Upload/select one
in Gradio if missing. The same persisted reference is passed to synthesis and its
full file SHA-256 defines `generated.voice` as `reference-sha256:<digest>`.
This identifier is stable across file moves and does not expose a local path.
`generated.tts` is the configured Chatterbox Python class; `generated.model` is
the actual configured checkpoint filename. Missing reference, model or
`[NOMBRE_CAPITULO]` title is an error, not a slug-derived fallback. No podcast
title, description, cover or publication status is inferred.

### Timing and gaps

The old mixer had no fixed silence between tagged blocks: transitions used
jingles of variable duration with a two-second overlap into the next voice.
The existing 0.25-second pauses are **inside** a voice block, between TTS chunks.
They are not chapter gaps or waits for quiz answers.

The new `generated.gap_sec` is the configured **additional silence** inserted
between each pair of theory tags, before any transition jingle. It is not the
total time between spoken blocks, a measured average, or a quiz-response timeout.
Jingle-only intervals remain variable. Silence is quantized to the nearest
sample. A value of `0` preserves the old transition behavior.

Each theory tag now gets its own manifest block, including all three introductory
tags, instead of grouping multiple tags/files into one introduction. Blocks have
consecutive orders starting at `1`. `script` is the source basename. Sample
offsets come from the finished mix after inserting silence, crossfading jingles,
and appending the closing jingle; division by sample rate retains decimal
precision. The closing jingle belongs to the last block.

Mutagen reads duration and bitrate from the **encoded final MP3**, not from a
requested bitrate or rounded WAV timings. SHA-256 covers the final MP3 bytes.
MP3 frame padding can extend the reported duration beyond the final mixed block.
Validation requires finite times, `0 <= start_sec < end_sec <= duration_sec`,
consecutive chronological orders and no overlapping blocks; gaps are valid.
The updated mixer is identified by `podcast_mix_version: 5`.

### Hashes and immutable versions

`generated.scripts_hash` is a full lowercase SHA-256. Its exact input is:

1. The bytes `chatterbox-scripts-v1` followed by one NUL byte.
2. All selected chapter source files, including quiz-only sources, sorted by
   basename using Python's case-sensitive Unicode string ordering.
3. For each file: the UTF-8 basename length as an unsigned 8-byte big-endian
   integer, the UTF-8 basename, the raw file content length in the same format,
   then the raw bytes (including any BOM and original line endings).

Absolute directory names, mtimes, audio and configuration are **not** included
in this source hash. Moving identical sources does not change it.

The generator adds `generated.request_hash` as a cache/publication extension
to the contract. It hashes canonical UTF-8 JSON (`ensure_ascii=False`,
`sort_keys=True`, separators `(",", ":")`, no NaN) containing the source hash,
TTS class, model, voice reference hash, gap, mix/schema versions, exaggeration,
temperature, seed, CFG weight, speed, requested bitrate, and full hashes of
the required music assets. It is not a claim that stochastic synthesis with
seed `0` is deterministic; only the final MP3 SHA-256 identifies its bytes.

An existing version is reused only when this complete request hash matches,
the manifest validates, the final MP3's measured metadata/hash matches, and all
expected quiz MP3s exist. A changed configuration, corrupt MP3, missing quiz file
or unverifiable manifest is an explicit error: choose a new version. Published
versions are never silently repaired or overwritten.

New audio and metadata are written in a private staging directory and the
complete directory is renamed to `v{N}` only after validation. Failed exports
keep cached WAVs and input scripts for retry; incomplete versions are not
published. After success, sources move to
`processed/{podcast}/{language}/ch{NN}/v{N}/`. To publish them again, move them
back into the input directory and choose the intended/new version.
Older unversioned outputs are left untouched.

There is **no R2 upload implementation** in this project. `r2_key` is a relative
object key for a future publisher; no credentials or local reference paths go
into the chapter manifest. The local version protections do not replace R2
conditional uploads if uploading is added later.

### Commands

In the existing Chatterbox environment, install the new declared dependency and
start the app:

```powershell
python -m pip install "mutagen>=1.47,<2"
python multilingual_app.py
```

Leave the main text empty, select a scripts directory, select the version and
gap, and provide the reference audio. The directory's filenames select each
chapter's language as before.

Run the focused tests without loading TTS models or launching Gradio:

```powershell
python -m unittest discover -s tests -v
```

MP3 metadata tests encode real PCM fixtures with lameenc and inspect the resulting
files with Mutagen. Publisher/TTS routing tests simulate synthesis, so these tests
do not certify model inference quality or playback interactions.
