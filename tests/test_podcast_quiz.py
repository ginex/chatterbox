import ast
import hashlib
import json
import re
import shutil
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock


APP_PATH = Path(__file__).resolve().parents[1] / "multilingual_app.py"


def load_audio_helpers():
    # Load the standalone app's helpers without launching Gradio or loading TTS models.
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    constants = {
        "PODCAST_BLOCK_TAGS", "PODCAST_BLOCK_PATTERN", "SCRIPT_NAME_PATTERN",
        "PROCESSED_SCRIPTS_DIRNAME", "INTRO_TAGS", "PODCAST_MIX_VERSION",
        "FINAL_JINGLE_FILENAME", "MP3_BITRATE_KBPS", "VOICE_TARGET_DBFS",
    }
    functions = {
        "parse_podcast_blocks", "split_podcast_blocks", "podcast_block_filename",
        "podcast_background_filename", "podcast_transition_filename",
        "required_podcast_assets", "mix_podcast_blocks", "export_quiz_audio",
        "synthesize_script", "generate_tts_audio", "list_script_files",
        "parse_script_name", "group_chapter_scripts", "read_script_file",
        "humanize_slug", "clean_title", "script_block_title", "move_to_processed",
        "load_json", "save_json",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in constants for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            node.decorator_list = []
            nodes.append(node)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    namespace = {
        "re": re, "Path": Path, "hashlib": hashlib, "json": json,
        "shutil": shutil, "datetime": datetime, "timezone": timezone, "time": time,
        "gr": SimpleNamespace(Error=ValueError, Progress=Mock, Warning=Mock(), update=Mock()),
    }
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


SCRIPT = """[NOMBRE_PODCAST] Podcast
[NOMBRE_CAPITULO] Chapter
[INTRO_CAPITULO] Introduction
[SECCION] Theory
[FINAL] Goodbye
[PREGUNTAS] Quiz introduction
[PREGUNTA_1] First question? A. One. B. Two.
[OPCION_CORRECTA_1] A
[RESPUESTA_1] Explanation one
[PREGUNTA_12] Another question?
[OPCION_CORRECTA_12] C
[RESPUESTA_12] Explanation twelve
[INCORRECTO] Try again
[CORRECTO] Well done
[ESPERA_1] Still waiting
[ESPERA_2] Answer now
[FIN_CAPITULO] The end
"""


class PodcastQuizTests(unittest.TestCase):
    def setUp(self):
        self.app = load_audio_helpers()
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_parser_separates_quiz_and_preserves_legacy_theory(self):
        blocks = self.app["parse_podcast_blocks"](SCRIPT.lower())
        theory, quiz = self.app["split_podcast_blocks"](blocks)
        self.assertEqual(len(theory), 5)
        self.assertEqual(theory[-1]["text"], "goodbye")
        self.assertEqual(len(quiz), 12)
        self.assertEqual(quiz[2]["text"], "a")
        self.assertEqual(quiz[4]["tag"], "PREGUNTA_12")
        self.assertEqual(quiz[-1]["tag"], "FIN_CAPITULO")
        legacy = self.app["parse_podcast_blocks"](SCRIPT.split("[PREGUNTAS]")[0])
        self.assertEqual(self.app["split_podcast_blocks"](legacy), (legacy, []))

    def test_ambiguous_quiz_layouts_raise_explicit_errors(self):
        for script in (
            "[SECCION] Theory [PREGUNTAS] Quiz",
            "[FINAL] Bye [PREGUNTA_1] One [PREGUNTA_1] Duplicate",
            "[FINAL] Bye [PREGUNTAS] Quiz [SECCION] More theory",
        ):
            with self.subTest(script=script), self.assertRaises(ValueError):
                self.app["split_podcast_blocks"](self.app["parse_podcast_blocks"](script))

    def test_mixer_repeats_final_transition_jingle_after_speech(self):
        blocks = self.app["parse_podcast_blocks"]("[SECCION] Theory [FINAL] Goodbye")
        voice = MagicMock()
        voice.shape = (2, 10)
        jingle = MagicMock()
        jingle.shape = (2, 6)
        jingle.__getitem__.return_value.__add__.return_value.abs.return_value.max.return_value = 0.5
        music_dir = self.root / "music"
        self.app.update({
            "PODCAST_MUSIC_DIR": music_dir,
            "as_stereo": lambda audio: audio,
            "load_audio": Mock(side_effect=lambda path, sr: jingle if path.name == "jingle2.wav" else voice),
            "mix_voice_with_ducked_music": Mock(side_effect=lambda audio, music, sr: audio),
            "torch": SimpleNamespace(cat=Mock()),
            "ta": SimpleNamespace(save=Mock(side_effect=lambda path, audio, sr: Path(path).touch())),
        })
        ranges = self.app["mix_podcast_blocks"](
            blocks, [self.root / "section.wav", self.root / "final.wav"],
            self.root / "combined.wav", 2,
        )
        self.assertEqual(ranges, [(0, 10), (12, 28)])
        parts = self.app["torch"].cat.call_args.args[0]
        self.assertIs(parts[-1], jingle)
        jingle_loads = [
            call for call in self.app["load_audio"].call_args_list
            if call.args[0] == music_dir / "jingle2.wav"
        ]
        self.assertEqual(len(jingle_loads), 2)
        final_only = self.app["parse_podcast_blocks"]("[FINAL] Goodbye")
        self.assertIn("jingle2.wav", self.app["required_podcast_assets"](final_only))

    def test_synthesis_routes_only_theory_to_mixer_and_keeps_quiz_voice(self):
        self.app.update({
            "CACHE_DIR": self.root,
            "LANGUAGE_MODELS": {"es": {"key": "es-es"}},
            "get_or_load_model": Mock(return_value=SimpleNamespace(sr=24000, generate=Mock(return_value=MagicMock()))),
            "split_text_into_chunks": lambda text: [text],
            "required_podcast_assets": Mock(return_value=set()),
            "persist_reference_audio": lambda path: path,
            "default_audio_for_ui": lambda language: None,
            "concatenate_audio_files": Mock(side_effect=lambda paths, output, sr: output.touch()),
            "ta": SimpleNamespace(save=Mock(side_effect=lambda path, audio, sr: Path(path).touch())),
        })

        def mix(blocks, paths, output, sr):
            output.touch()
            return [(index, index + 1) for index in range(len(blocks))]

        self.app["mix_podcast_blocks"] = Mock(side_effect=mix)
        synthesize = self.app["synthesize_script"]
        combined = synthesize(SCRIPT, "es", remember_request=False)
        mixed_blocks, mixed_paths, *_ = self.app["mix_podcast_blocks"].call_args.args
        self.assertEqual([block["tag"] for block in mixed_blocks], list(self.app["PODCAST_BLOCK_TAGS"]))
        self.assertEqual(len(mixed_paths), 5)
        manifest = self.app["load_json"](combined.parent / "manifest.json")
        self.assertEqual(len(manifest["segments_sec"]), 5)
        self.assertEqual(len(manifest["quiz_outputs"]), 12)
        self.assertTrue(all(Path(output["path"]).is_file() for output in manifest["quiz_outputs"]))
        self.assertEqual(self.app["get_or_load_model"].return_value.generate.call_count, 17)
        # A missing quiz WAV must prevent recovery of a supposedly complete job.
        Path(manifest["quiz_outputs"][0]["path"]).unlink()
        self.app["ta"].load = Mock(return_value=(None, 24000))
        synthesize(SCRIPT, "es", remember_request=False)
        self.assertTrue(Path(manifest["quiz_outputs"][0]["path"]).is_file())
        self.assertEqual(self.app["mix_podcast_blocks"].call_count, 2)

    def configure_publisher(self):
        scripts_dir = self.root / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "demo__es__c01__chapter.txt"
        script.write_text(SCRIPT, encoding="utf-8")
        self.app.update({
            "PUBLISH_DIR": self.root / "publish",
            "LANGUAGE_MODELS": {"es": {"key": "es-es"}},
            "t3_model_filename": lambda language: "model.safetensors",
        })

        def synthesize(text, *args, **kwargs):
            job_dir = self.root / "job"
            job_dir.mkdir(exist_ok=True)
            combined = job_dir / "combined.wav"
            combined.write_bytes(b"theory")
            theory, quiz = self.app["split_podcast_blocks"](self.app["parse_podcast_blocks"](text))
            outputs = []
            for block in quiz:
                wav = job_dir / f"{block['tag']}.wav"
                wav.write_bytes(str(block["text"]).encode())
                outputs.append({"tag": block["tag"], "path": str(wav)})
            self.app["save_json"](job_dir / "manifest.json", {
                "segments_sec": [[index * 10, (index + 1) * 10] for index in range(len(theory))],
                "quiz_outputs": outputs,
            })
            return combined

        self.app["synthesize_script"] = Mock(side_effect=synthesize)
        self.app["convert_wav_to_mp3"] = Mock(side_effect=lambda wav, mp3: mp3.write_bytes(wav.read_bytes()))
        return scripts_dir, script

    def test_publishing_outputs_metadata_and_repairs_missing_quiz_files(self):
        scripts_dir, script = self.configure_publisher()
        publish = self.app["generate_tts_audio"]
        _, result = publish("", str(scripts_dir), None, "es")
        chapter_dir = self.app["PUBLISH_DIR"] / "demo" / "es" / "ch01"
        metadata = self.app["load_json"](chapter_dir / "ch01.json")
        self.assertEqual(len(result.splitlines()), 13)
        self.assertEqual(len(metadata["quiz"]), 12)
        self.assertEqual(len(metadata["blocks"]), 3)
        self.assertEqual(metadata["audio"]["duration_sec"], 50)
        for entry in metadata["quiz"]:
            self.assertTrue((self.app["PUBLISH_DIR"] / entry["r2_key"]).is_file())
            self.assertEqual(Path(entry["r2_key"]).stem, entry["tag"].lower())
        self.assertEqual((chapter_dir / "quiz" / "opcion_correcta_1.mp3").read_bytes(), b"A")
        self.assertEqual((chapter_dir / "quiz" / "fin_capitulo.mp3").read_bytes(), b"The end")
        processed = scripts_dir / "processed" / "demo" / "es" / "ch01" / script.name
        self.assertTrue(processed.is_file())
        self.assertFalse((self.root / "job").exists())

        processed.replace(script)
        publish("", str(scripts_dir), None, "es")
        self.assertEqual(self.app["synthesize_script"].call_count, 1)

        processed.replace(script)
        (chapter_dir / "quiz" / "pregunta_12.mp3").unlink()
        publish("", str(scripts_dir), None, "es")
        self.assertEqual(self.app["synthesize_script"].call_count, 2)
        self.assertTrue((chapter_dir / "quiz" / "pregunta_12.mp3").is_file())

    def test_export_failure_keeps_scripts_and_cached_audio_for_retry(self):
        scripts_dir, script = self.configure_publisher()
        encoder = self.app["convert_wav_to_mp3"].side_effect

        def fail_on_quiz(wav, mp3):
            if mp3.parent.name == "quiz":
                raise OSError("Encoder failed")
            return encoder(wav, mp3)

        self.app["convert_wav_to_mp3"].side_effect = fail_on_quiz
        with self.assertRaisesRegex(OSError, "Encoder failed"):
            self.app["generate_tts_audio"]("", str(scripts_dir), None, "es")
        self.assertTrue(script.is_file())
        self.assertTrue((self.root / "job" / "combined.wav").is_file())
        self.assertFalse(list(self.app["PUBLISH_DIR"].rglob("*.json")))

    def test_multifile_chapter_keeps_quiz_out_of_theory_metadata(self):
        scripts_dir, script = self.configure_publisher()
        intro, body = SCRIPT.split("[SECCION]", 1)
        theory, quiz = ("[SECCION]" + body).split("[PREGUNTAS]", 1)
        script.unlink()
        for index, content in enumerate((intro, theory, "[PREGUNTAS]" + quiz), start=1):
            path = scripts_dir / f"demo__es__c01-b{index:02d}__block.txt"
            path.write_text(content, encoding="utf-8")
        self.app["generate_tts_audio"]("", str(scripts_dir), None, "es")
        metadata = self.app["load_json"](self.app["PUBLISH_DIR"] / "demo" / "es" / "ch01" / "ch01.json")
        self.assertEqual(len(metadata["blocks"]), 2)
        self.assertEqual(metadata["blocks"][-1]["start_sec"], 30)
        self.assertEqual(metadata["blocks"][-1]["end_sec"], 50)
        self.assertEqual(len(metadata["quiz"]), 12)

    def test_text_mode_exports_quiz_but_preserves_theory_wav(self):
        self.configure_publisher()
        combined, result = self.app["generate_tts_audio"](SCRIPT, "", None, "es")
        self.assertEqual(Path(combined).read_bytes(), b"theory")
        outputs = [Path(path) for path in result.splitlines()]
        self.assertEqual(len(outputs), 12)
        self.assertTrue(all(path.parent == Path(combined).parent / "quiz" for path in outputs))
        self.assertTrue((Path(combined).parent / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
