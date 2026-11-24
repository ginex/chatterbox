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

Directory generation keeps `publish/{podcast}/{idioma}/ch{NN}/ch{NN}.mp3`
for theory only, from `[NOMBRE_PODCAST]` through `[FINAL]` and the closing jingle.
Each quiz block is exported next to it in `quiz/`, using the lowercase tag as
the filename, for example `quiz/pregunta_1.mp3`, `quiz/opcion_correcta_1.mp3`,
and `quiz/fin_capitulo.mp3`. These files contain only normalized speech:
no background music, jingles, or crossfades. The chapter JSON lists their tags
and R2 keys under `quiz`; theory block timings exclude quiz audio.

In text-input mode the theory output remains a WAV, with quiz MP3s under the
job's `quiz/` directory. Generated MP3 paths are shown in the UI in both modes.
Existing chapter outputs are reused only when the script hash and mix version
match and all expected quiz MP3s exist. Source scripts are moved to `processed/`
only after all chapter audio and metadata have been written.
To regenerate a chapter already in `processed/`, move its scripts back into
the input directory.
