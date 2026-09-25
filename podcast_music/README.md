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
