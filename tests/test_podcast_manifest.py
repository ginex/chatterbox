import array
import copy
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import lameenc

from podcast_manifest import (
    build_chapter_manifest,
    mp3_metadata,
    positive_version,
    quiz_metadata,
    reference_voice_id,
    request_hash,
    scripts_hash,
    validate_chapter_manifest,
    validate_gap,
)


class ChapterManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.txt"
        self.source.write_bytes(b"[NOMBRE_CAPITULO] Test\n[FINAL] End\n")
        self.reference = self.root / "reference.wav"
        self.reference.write_bytes(b"test-only reference fixture")
        self.mp3 = self.root / "ch02.mp3"
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(64)
        encoder.set_in_sample_rate(24000)
        encoder.set_channels(1)
        samples = array.array("h", (int(2000 * math.sin(index / 10)) for index in range(24000)))
        self.mp3.write_bytes(encoder.encode(samples.tobytes()) + encoder.flush())
        self.generated = {
            "tts": "tests.PcmFixture",
            "model": "sine-pcm-24000",
            "voice": reference_voice_id(str(self.reference)),
            "gap_sec": 0.125,
            "scripts_hash": scripts_hash([self.source]),
            "podcast_mix_version": 5,
        }
        self.generated["request_hash"] = request_hash(self.generated)
        self.blocks = [
            {"order": 1, "title": "Introduction", "script": self.source.name, "start_sec": 0, "end_sec": 0.3333333333333333},
            {"order": 2, "title": "End", "script": self.source.name, "start_sec": 0.5, "end_sec": 1.0},
        ]
        texts = {
            "PREGUNTAS": "Questions.",
            "PREGUNTA_1": "Which one? A. First. B. Second. D. Fourth.",
            "OPCION_CORRECTA_1": "B",
            "RESPUESTA_1": "B is the second.",
            "INCORRECTO": "Incorrect.",
            "CORRECTO": "Correct.",
            "ESPERA_1": "Still waiting.",
            "ESPERA_2": "Your answer?",
            "FIN_CAPITULO": "End.",
        }
        self.quiz_blocks = [{"tag": tag, "text": text} for tag, text in texts.items()]

    def build(self):
        return build_chapter_manifest(
            podcast="demo", language="fr", chapter=2, version=3, title="Test",
            mp3_path=self.mp3, blocks=copy.deepcopy(self.blocks), quiz=quiz_metadata(self.quiz_blocks),
            generated=self.generated,
        )

    def test_valid_manifest_uses_final_mp3_and_versioned_keys(self):
        manifest = self.build()
        validate_chapter_manifest(manifest)
        self.assertEqual(manifest["audio"]["sha256"], hashlib.sha256(self.mp3.read_bytes()).hexdigest())
        self.assertEqual(manifest["audio"]["bitrate_kbps"], 64)
        self.assertGreater(manifest["audio"]["duration_sec"], 1.0)
        self.assertEqual(manifest["audio"]["r2_key"], "demo/fr/ch02/v3/ch02.mp3")
        self.assertEqual([block["order"] for block in manifest["blocks"]], [1, 2])
        self.assertEqual(manifest["blocks"][0]["end_sec"], 0.3333333333333333)
        self.assertTrue(all(entry["r2_key"].startswith("demo/fr/ch02/v3/quiz/") for entry in manifest["quiz"]))
        self.assertEqual(manifest["quiz"][1]["option_letters"], ["A", "B", "D"])
        self.assertEqual(manifest["quiz"][2]["correct_option"], "B")
        self.assertTrue(manifest["generated"]["date"].endswith("Z"))

    def test_invalid_times_overlaps_and_orders_are_rejected_but_gaps_are_valid(self):
        valid = self.build()
        validate_chapter_manifest(valid)
        for changes in (
            {"start_sec": -0.1},
            {"end_sec": 0},
            {"end_sec": valid["audio"]["duration_sec"] + 1},
            {"start_sec": float("nan")},
            {"end_sec": float("inf")},
            {"order": 0},
        ):
            manifest = copy.deepcopy(valid)
            manifest["blocks"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_chapter_manifest(manifest)
        manifest = copy.deepcopy(valid)
        manifest["blocks"][1]["start_sec"] = 0.2
        with self.assertRaisesRegex(ValueError, "Solapamiento"):
            validate_chapter_manifest(manifest)
        manifest = copy.deepcopy(valid)
        manifest["blocks"][1]["order"] = 1
        with self.assertRaisesRegex(ValueError, "consecutivamente desde 1"):
            validate_chapter_manifest(manifest)

    def test_source_hash_is_complete_stable_and_includes_names_and_raw_bytes(self):
        second = self.root / "second.txt"
        second.write_bytes(b"Additional source\r\n")
        expected = scripts_hash([self.source, second])
        self.assertEqual(len(expected), 64)
        self.assertEqual(expected, scripts_hash([second, self.source]))
        relocated = self.root / "moved"
        relocated.mkdir()
        copies = []
        for source in (self.source, second):
            target = relocated / source.name
            target.write_bytes(source.read_bytes())
            copies.append(target)
        self.assertEqual(expected, scripts_hash(copies))
        second.write_bytes(b"Additional source\n")
        self.assertNotEqual(expected, scripts_hash([self.source, second]))
        renamed = self.source.rename(self.root / "renamed.txt")
        self.assertNotEqual(scripts_hash([copies[0]]), scripts_hash([renamed]))

    def test_missing_required_generation_configuration_fails_explicitly(self):
        valid = self.build()
        for field in ("tts", "voice", "model"):
            for value in (None, "", "unknown", "default"):
                manifest = copy.deepcopy(valid)
                manifest["generated"][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    validate_chapter_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "Falta la voz"):
            reference_voice_id(None)
        with self.assertRaisesRegex(ValueError, "Falta la voz"):
            reference_voice_id(str(self.root / "missing.wav"))

    def test_invalid_version_gap_and_source_paths_fail(self):
        for value in (0, -1, 1.5, True, float("inf"), "2"):
            with self.subTest(version=value), self.assertRaises(ValueError):
                positive_version(value)
        for value in (-1, float("inf"), float("nan"), True, "2"):
            with self.subTest(gap=value), self.assertRaises(ValueError):
                validate_gap(value)
        manifest = self.build()
        manifest["blocks"][0]["script"] = r"C:\private\source.txt"
        with self.assertRaisesRegex(ValueError, "archivo fuente"):
            validate_chapter_manifest(manifest)
        manifest = self.build()
        manifest["quiz"][0]["r2_key"] = "demo/fr/ch02/quiz/preguntas.mp3"
        with self.assertRaisesRegex(ValueError, "Ruta de quiz"):
            validate_chapter_manifest(manifest)

    def test_quiz_requires_readable_options_and_single_valid_correct_letter(self):
        for text in (
            "Which? A) First. B) Second.",
            "Which?\n(A) First.\n(B) Second.",
            "Which? A: First. B: Second.",
        ):
            blocks = copy.deepcopy(self.quiz_blocks)
            blocks[1]["text"] = text
            blocks[2]["text"] = " b "
            metadata = quiz_metadata(blocks)
            self.assertEqual(metadata[1]["option_letters"], ["A", "B"])
            self.assertEqual(metadata[2]["correct_option"], "B")
        for correct in ("Z", "B.", "BC", ""):
            blocks = copy.deepcopy(self.quiz_blocks)
            blocks[2]["text"] = correct
            with self.subTest(correct=correct), self.assertRaises(ValueError):
                quiz_metadata(blocks)
        blocks = copy.deepcopy(self.quiz_blocks)
        blocks[1]["text"] = "No labelled alternatives."
        with self.assertRaisesRegex(ValueError, "alternativas"):
            quiz_metadata(blocks)
        with self.assertRaisesRegex(ValueError, "incompleto"):
            quiz_metadata(self.quiz_blocks[:-1])

    def test_invalid_audio_is_not_accepted_as_an_mp3(self):
        self.mp3.write_bytes(b"not an mp3")
        with self.assertRaisesRegex(ValueError, "MP3 final no valido"):
            mp3_metadata(self.mp3)


if __name__ == "__main__":
    unittest.main()
