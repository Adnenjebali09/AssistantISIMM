import os
import tempfile
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from isimm.audio import Frontend, Turn, Microphone
from isimm.config import Settings
from isimm.conversation import Session, finished
from isimm.feedback import Feedback
from isimm.knowledge import Library
from isimm.llm import Ollama, select_models
from isimm.stt import Transcript, Whisper
from isimm.tts import spoken, Speech
from isimm.vision import GazeTimer, geometry
from scripts.evaluate import errors


class GazeTests(unittest.TestCase):
    def test_five_seconds_fire_once(self):
        timer = GazeTimer()
        fires = [i for i in range(90) if timer.update(i / 10, (0.5, 0.5))[1]]
        self.assertEqual(fires, [50])

    def test_gap_breaks_hold(self):
        timer = GazeTimer()
        for i in range(49):
            timer.update(i / 10, (0.5, 0.5))
        timer.update(5.2, None)
        self.assertEqual(timer.update(5.3, (0.5, 0.5)), (0.0, False))

    def test_short_gap_pauses(self):
        timer = GazeTimer()
        for i in range(10):
            timer.update(i / 10, (0.5, 0.5))
        timer.update(1, None)
        self.assertAlmostEqual(timer.update(1.1, (0.5, 0.5))[0], 0.9 / 5)
        self.assertEqual(timer.update(1.2, (0.75, 0.5))[0], 0)

    def test_stalled_stream_does_not_credit_missing_frames(self):
        timer = GazeTimer()
        timer.update(0, (0.5, 0.5))
        self.assertEqual(timer.update(8, (0.5, 0.5)), (0.0, False))

    def test_roll_calibration(self):
        from math import sin, cos, pi

        pts = [(0.5, 0.5)] * 478
        for i, x, y in (
            (234, 0.2, 0.5),
            (454, 0.8, 0.5),
            (10, 0.5, 0.2),
            (152, 0.5, 0.8),
            (33, 0.3, 0.45),
            (133, 0.4, 0.45),
            (159, 0.35, 0.43),
            (145, 0.35, 0.47),
            (468, 0.35, 0.45),
            (362, 0.6, 0.45),
            (263, 0.7, 0.45),
            (386, 0.65, 0.43),
            (374, 0.65, 0.47),
            (473, 0.65, 0.45),
        ):
            pts[i] = (x, y)
        for angle in (0, pi / 2, -pi / 3):
            face = [
                NS(
                    x=0.5 + cos(angle) * (x - 0.5) - sin(angle) * (y - 0.5),
                    y=0.5 + sin(angle) * (x - 0.5) + cos(angle) * (y - 0.5),
                )
                for x, y in pts
            ]
            self.assertIsNotNone(geometry([face], Settings())[1])

    def test_multiple_faces(self):
        face = [NS(x=0.5, y=0.4)]
        self.assertIsNone(geometry([face, face], Settings())[1])


class AudioTests(unittest.TestCase):
    def test_pause_and_endpoint(self):
        turn = Turn(0.6)
        frame = np.zeros(512, np.float32)
        for p in [0] * 10 + [0.9] * 12 + [0] * 13 + [0.8] * 10 + [0] * 27:
            self.assertFalse(turn.push(frame, p))
        self.assertTrue(turn.push(frame, 0))
        self.assertGreater(len(turn.samples()), 25 * 512)

    def test_short_and_empty(self):
        for n in (0, 1, 4):
            turn = Turn(0.6)
            for p in [0.9] * n + [0] * 30:
                turn.push(np.zeros(512), p)
            self.assertIsNone(turn.samples())

    def test_fir_keeps_state(self):
        signal = np.random.default_rng(2).normal(size=3072).astype("float32")
        front = Frontend()
        self.assertTrue(
            np.allclose(
                np.concatenate(
                    (front.process(signal[:1536]), front.process(signal[1536:]))
                ),
                Frontend().process(signal),
                atol=1e-6,
            )
        )

    def test_microphone_failure_releases_indicator(self):
        cfg = Settings(denoise=False)
        events = Mock()
        mic = Microphone(cfg, events)
        mic.vad = Mock()
        sd = NS(PortAudioError=OSError, InputStream=Mock(side_effect=OSError("absent")))
        with patch.dict("sys.modules", sounddevice=sd):
            with self.assertRaisesRegex(RuntimeError, "Microphone"):
                mic.listen(threading.Event())
        events.assert_any_call("mic", False)


class EngineTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("ISIMM_TEST_LLM") == "1",
        "Essai facultatif avec Ollama local et Qwen 4B",
    )
    def test_real_followup_keeps_reference(self):
        bot = Ollama(Settings(llm="qwen3.5:4b"), lambda *_: None)
        history = []
        for question in (
            "Quels départements comporte l’ISIMM ?",
            "En quelle année cet arrêté a-t-il été publié ?",
        ):
            reply = "".join(bot.stream(question, "fr", history, threading.Event()))
            if history:
                self.assertIn("2022", reply)
            else:
                for term in ("technolog", "informatique", "math"):
                    self.assertIn(term, reply.casefold())
            history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": reply},
                ]
            )

    def test_knowledge_and_unknown(self):
        self.assertEqual(Library().search("les départements")[0]["status"], "verified")
        self.assertEqual(
            Library().search("emploi du temps")[0]["status"], "unavailable"
        )
        self.assertFalse(Library().search("xyz"))

    def test_model_budget(self):
        choices = ["qwen3.5:4b", "qwen3.5:2b"]
        self.assertEqual(
            select_models("auto", choices, dict(vram_free=0, available=1), False), []
        )
        self.assertEqual(
            select_models("auto", choices, dict(vram_free=3200, available=1), False),
            choices[1:],
        )
        self.assertEqual(
            select_models("auto", choices, dict(vram_free=5500, available=4), False),
            choices,
        )

    def test_cpu_fallback(self):
        with patch(
            "isimm.stt.resources", return_value=dict(vram_free=0, available=0.8)
        ):
            self.assertEqual(
                list(Whisper(Settings(), Mock()).candidates()),
                [("base", "cpu", "int8")],
            )

    def test_ollama_down(self):
        with patch(
            "http.client.HTTPConnection.request", side_effect=ConnectionRefusedError()
        ):
            with self.assertRaisesRegex(RuntimeError, "Ollama"):
                list(
                    Ollama(Settings(), Mock()).stream(
                        "Bonjour", "fr", [], threading.Event()
                    )
                )

    def test_privacy_and_retention(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Feedback(1, tmp)
            self.assertFalse(f.add("idée"))
            self.assertTrue(f.add("idée", True))
            with patch("isimm.feedback.time.time", return_value=1e12):
                f.prune()
            self.assertEqual(
                f.db.execute("select count(*) from suggestion").fetchone()[0], 0
            )
            f.db.close()

    def test_quality_metrics(self):
        self.assertEqual(errors("École, ici !", "ecole ici"), (0, 0))
        self.assertEqual(errors("un deux", "un trois")[0], 0.5)

    def test_voice_cleanup(self):
        self.assertEqual(
            spoken("**Bonjour** [campus](https://x) ```python print(1)```"),
            "Bonjour campus",
        )
        self.assertTrue(finished("Merci, au revoir !"))
        self.assertFalse(finished("Comment dire au revoir en arabe ?"))

    def test_optional_cloud(self):
        speech = Speech(Settings(), Mock(), Mock(side_effect=AssertionError))
        speech.azure = Mock(side_effect=AssertionError)
        with patch("piper.PiperVoice.load", side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):
                list(speech.render("مرحبا", "ar"))
        speech.azure.assert_not_called()


class ConversationTests(unittest.TestCase):
    def test_full_cycle_echo_gate_and_history(self):
        stop = threading.Event()
        events = []

        def emit(name, value):
            events.append((name, value))

        mic, stt, llm, tts = Mock(), Mock(), Mock(), Mock()
        mic.last_speech = None
        mic.listen.side_effect = [np.ones(512)] * 3
        stt.transcribe.side_effect = [
            Transcript("Question", "fr", 1, False),
            Transcript("Et ensuite ?", "fr", 1, False),
            Transcript("Bye", "en", 1, False),
        ]
        contexts = []

        def stream(q, lang, history, stop):
            contexts.append(list(history))
            yield "Réponse. "
            yield "Une précision."

        llm.stream.side_effect = stream

        def say(*args):
            self.assertIn(
                [v for k, v in events if k == "state"][-1], ["GREETING", "SPEAKING"]
            )
            if len(args) > 3:
                args[3]()

        tts.say.side_effect = say
        Session(mic, stt, llm, tts, emit).run(stop)
        self.assertEqual(len(contexts[1]), 2)
        self.assertEqual(events[-1], ("state", "END_CONVERSATION"))
        self.assertEqual(sum(k == "timing" for k, v in events), 2)

    def test_uncertain_then_silence(self):
        mic, stt, llm, tts = Mock(), Mock(), Mock(), Mock()
        mic.last_speech = None
        mic.listen.side_effect = [np.ones(512), None]
        stt.transcribe.return_value = Transcript("???", "ar", 0.4, True)
        Session(mic, stt, llm, tts, Mock()).run(threading.Event())
        llm.stream.assert_not_called()
        self.assertEqual(tts.say.call_count, 2)
