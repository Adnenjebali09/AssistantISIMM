import os
import time
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox
from isimm.config import Settings
from isimm.feedback import Feedback
from isimm.gui import Window


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_stop_clears_text_and_controls(self):
        with TemporaryDirectory() as tmp:
            with patch(
                "isimm.gui.Feedback", side_effect=lambda days: Feedback(days, tmp)
            ):
                win = Window(Settings())
            try:
                with patch.object(
                    QMessageBox, "question", return_value=QMessageBox.Yes
                ):
                    win.start()
                self.assertTrue(win.talk.isEnabled())
                win.handle(
                    "transcript",
                    dict(
                        text="Une question ?", lang="fr", seconds=0.5, uncertain=False
                    ),
                )
                self.assertEqual(win.question.toPlainText(), "Une question ?")
                win.shutdown()
                self.assertFalse(win.question.toPlainText())
                self.assertFalse(win.talk.isEnabled())
                self.assertTrue(win.start_btn.isEnabled())
            finally:
                win.close()

    @unittest.skipUnless(
        os.getenv("ISIMM_TEST_VOICE") == "1",
        "Essai facultatif sur le micro et la sortie du bureau",
    )
    def test_manual_greeting_opens_microphone_and_stops(self):
        with TemporaryDirectory() as tmp:
            with patch(
                "isimm.gui.Feedback", side_effect=lambda days: Feedback(days, tmp)
            ):
                win = Window(Settings(voice="piper", volume=0.15))
            seen = []
            original = win.handle

            def observe(key, value):
                if key in ("mic", "state"):
                    seen.append((key, value))
                original(key, value)

            win.handle = observe
            try:
                with patch.object(
                    QMessageBox, "question", return_value=QMessageBox.Yes
                ):
                    win.start_btn.click()
                win.talk.click()
                worker = win.voice
                win.activate(True)
                self.assertIs(worker, win.voice)
                deadline = time.monotonic() + 25
                while ("mic", True) not in seen and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.02)
                self.assertIn(("mic", True), seen, win.notice.text())
                self.assertLess(
                    seen.index(("state", "GREETING")),
                    seen.index(("state", "LISTENING")),
                )
                self.assertLess(
                    seen.index(("state", "LISTENING")), seen.index(("mic", True))
                )
                win.end.click()
                self.assertEqual(win.machine.state.value, "COOLDOWN")
                self.assertFalse(worker.is_alive())
            finally:
                win.close()
