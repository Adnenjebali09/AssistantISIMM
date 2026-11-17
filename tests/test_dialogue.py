"""Tests de la chaîne conversationnelle avec un transport simulé ou Ollama local."""

import io
import json
import os
import re
import threading
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from isimm.config import Settings
from isimm.conversation import Session
from isimm.dialogue import (
    DialogueTrace,
    recent_messages,
    reply_locale,
    comparison_reference,
    followup_context,
)
from isimm.knowledge import Library
from isimm.llm import Ollama, needs_records, fact_literals
from isimm.planning import time_budget, render_plan
from isimm.stt import Transcript
from isimm.tts import spoken
from scripts.evaluate_dialogue import evaluate


def response(data):
    body = io.BytesIO(data.encode())
    body.status = 200
    return body


class DialogueTests(unittest.TestCase):
    def client(self, *replies):
        configured = "qwen3.5:2b"
        conn = Mock()
        conn.getresponse.side_effect = [
            response(json.dumps({"models": [{"name": configured}]})),
            response('{"models": []}'),
            *(
                response(json.dumps({"message": {"content": r}, "done": True}) + "\n")
                for r in replies
            ),
        ]
        self.enterContext(
            patch("isimm.llm.http.client.HTTPConnection", return_value=conn)
        )
        self.enterContext(
            patch(
                "isimm.llm.resources",
                return_value={"vram_total": 6144, "vram_free": 5000, "available": 4},
            )
        )
        trace = DialogueTrace(True)
        return Ollama(Settings(llm=configured), Mock(), trace=trace), conn

    def test_review_to_wire_and_spoken_diagnostics(self):
        bot, conn = self.client("**تنجم** تراجع.")
        mic, stt, tts = Mock(), Mock(), Mock()
        mic.listen.side_effect = [object(), None]
        mic.last_speech = None
        stt.transcribe.return_value = Transcript("texte mal compris", "fr", 0.7, False)
        question = "كيفاش نراجع؟"
        trace = io.StringIO()
        session = Session(
            mic, stt, bot, tts, Mock(), review=lambda _: question, trace=bot.trace
        )
        with redirect_stderr(trace):
            session.run(threading.Event())
        stages = [json.loads(line) for line in trace.getvalue().splitlines()]
        self.assertEqual(
            [r["dialogue"] for r in stages],
            [
                "stt",
                "reviewed_transcript",
                "retrieval",
                "ollama_request",
                "ollama_raw",
                "tts_final",
            ],
        )
        self.assertEqual(stages[0]["text"], "texte mal compris")
        self.assertEqual(stages[1]["text"], question)
        self.assertEqual(len({r["turn"] for r in stages}), 1)
        body = json.loads(conn.request.call_args.args[2])
        self.assertEqual(body, stages[3]["body"])
        self.assertEqual(body["messages"][-1], {"role": "user", "content": question})
        self.assertEqual(stages[4]["content"], "**تنجم** تراجع.")
        sent, lang = tts.say.call_args.args[:2]
        self.assertEqual(
            (stages[-1]["text"], stages[-1]["language"]), (spoken(sent), lang)
        )
        self.assertEqual(lang, "ar")

    def test_unsupported_answer_never_reaches_listener(self):
        bot, _ = self.client("L’examen commence à 18h45.", '"unsupported"')
        with redirect_stderr(io.StringIO()):
            received = list(
                bot.stream(
                    "Quel est mon horaire d’examen ?", "fr", [], threading.Event()
                )
            )
        self.assertEqual(len(received), 1)
        self.assertNotIn("18h45", received[0])
        self.assertIn("temps", received[0])

    def test_study_question_with_campus_keywords_can_be_answered(self):
        original = "Comment préparer un examen de maths ?"
        candidate = "Refais un exercice corrigé."
        bot, conn = self.client(candidate)
        with redirect_stderr(io.StringIO()):
            result = "".join(bot.stream(original, "fr", [], threading.Event()))
        self.assertEqual(result, candidate)
        for call in conn.request.call_args_list:
            if call.args[0] == "POST":
                self.assertEqual(
                    json.loads(call.args[2])["messages"][-1]["content"], original
                )

    def test_memory_scope_and_retrieval(self):
        library = Library()
        history = [
            {"role": "user", "content": "Quels départements comporte l’ISIMM ?"},
            {"role": "assistant", "content": "Trois départements."},
        ]
        self.assertIn(
            "departements", [r["id"] for r in library.retrieve("Et combien ?", history)]
        )
        self.assertFalse(
            library.retrieve(
                "Comment organiser mon temps pour réviser deux matières ?", history
            )
        )
        self.assertFalse(
            library.retrieve("Changeons de sujet. Et une pile en Python ?", history)
        )
        messages = [
            {"role": role, "content": str(i) * 700}
            for i in range(5)
            for role in ("user", "assistant")
        ]
        saved = recent_messages(messages)
        self.assertEqual(
            recent_messages(
                history + [{"role": "user", "content": "Autre sujet. Mon CV."}]
            ),
            [{"role": "user", "content": "Autre sujet. Mon CV."}],
        )
        self.assertEqual(len(saved), 6)
        self.assertTrue(saved[0]["content"].startswith("2"))
        self.assertLessEqual(max(len(m["content"]) for m in saved), 650)

    def test_campus_and_general_intents(self):
        lib = Library()
        for q in (
            "Quel est le site web de l’ISIMM ?",
            "Quels documents faut-il pour mon inscription ?",
            "ما هي أقسام المعهد؟",
            "Qui est le directeur ?",
        ):
            self.assertTrue(needs_records(q, lib.search(q)), q)
        for q in (
            "Rédige un message à la scolarité.",
            "Comment préparer un examen de maths ?",
            "Que mettre dans un CV pour un stage ?",
        ):
            self.assertFalse(needs_records(q, lib.search(q)), q)
        self.assertEqual(
            fact_literals("Trois en ٢٠٢٢, voir www.isimm.rnu.tn."),
            {"3", "2022", "isimm.rnu.tn"},
        )

    def test_followup_drops_older_unrelated_topic(self):
        history = [
            {"role": "user", "content": "Quel est le site de l’ISIMM ?"},
            {"role": "assistant", "content": "Un site."},
            {
                "role": "user",
                "content": "Comment réviser deux matières cette semaine ?",
            },
            {"role": "assistant", "content": "Prévoir des séances."},
        ]
        self.assertEqual(
            followup_context(history, "Comment les répartir ?"), history[2:]
        )

    def test_comparison_reference(self):
        for comparison, followup, term in [
            (
                "Quelle différence entre un capteur et un moteur ?",
                "Un exemple pour le second ?",
                "un moteur",
            ),
            (
                "شنوّة الفرق بين pile و file في informatique؟",
                "عطيني مثال على الثانية.",
                "file",
            ),
            (
                "What is the difference between RAM and storage?",
                "Who invented the latter?",
                "storage",
            ),
        ]:
            self.assertEqual(
                comparison_reference(
                    [{"role": "user", "content": q} for q in (comparison, followup)]
                ),
                term,
            )

    def test_next_student_starts_without_history(self):
        mic, stt, llm, tts = Mock(), Mock(), Mock(), Mock()
        mic.last_speech = None
        mic.listen.side_effect = [object(), None, object(), None]
        stt.transcribe.side_effect = [
            Transcript(q, "fr", 0.2, False)
            for q in ("Une question", "Nouvelle question")
        ]
        session = Session(mic, stt, llm, tts, Mock())
        snapshots = []

        def stream(q, lang, history, stop):
            snapshots.append(tuple(history))
            return iter(["Réponse."])

        llm.stream.side_effect = stream
        session.run(threading.Event())
        session.run(threading.Event())
        self.assertEqual(snapshots, [(), ()])

    def test_time_constraints(self):
        for phrase, expected in [
            ("Comment répartir mes 47 minutes ?", 47),
            ("How to split half an hour?", 30),
            ("نص ساعة كيفاش نقسم وقتي؟", 30),
            ("Comment répartir deux heures et trente minutes ?", None),
            ("À quelle heure est le cours ?", None),
        ]:
            self.assertEqual(time_budget(phrase), expected)
        labels = {
            "activities": [
                {"activity": s, "weight": w}
                for s, w in [("Français", 3), ("Physique", 2), ("Pause", 1)]
            ]
        }
        answer = render_plan(json.dumps(labels), 47, "fr")
        self.assertEqual(sum(map(int, re.findall(r"\d+", answer))), 47)
        self.assertIn("Physique", answer)
        self.assertIsNone(render_plan("incomplet", 30, "ar"))

    def test_register_and_private_default(self):
        q = "شنوّة معناها RAM ؟"
        self.assertEqual(reply_locale(q, "fr", []), "ar-TN")
        self.assertEqual(reply_locale("أريد معرفة أقسام المعهد", "fr", []), "ar")
        previous = [{"role": "user", "content": q}]
        self.assertEqual(reply_locale("أعطني مثالا", "ar", previous), "ar-TN")
        self.assertEqual(reply_locale("اشرح بالفصحى", "ar", previous), "ar")
        self.assertEqual(reply_locale("Pourquoi ?", "fr", previous), "fr")
        output = io.StringIO()
        with redirect_stderr(output):
            trace = DialogueTrace()
            trace.begin()
            trace.log("stt", text=q)
        self.assertEqual(output.getvalue(), "")

    @unittest.skipUnless(
        os.environ.get("ISIMM_TEST_DIALOGUE") == "1",
        "Corpus multilingue avec Ollama local",
    )
    def test_real_conversations(self):
        results = list(evaluate(os.environ.get("ISIMM_TEST_MODEL", "qwen3.5:4b")))
        unique = {r["reply"].strip().casefold() for r in results}
        self.assertGreaterEqual(len({r["chat"] for r in results}), 15)
        self.assertGreaterEqual(len(unique), len(results) - 2)
        misses = [r["question"] for r in results if r["missing"]]
        self.assertGreaterEqual(
            (len(results) - len(misses)) / len(results), 0.85, misses
        )
        for row in results:
            limit = time_budget(row["question"])
            if limit:
                durations = re.findall(r"\((\d+) (?:minutes?|دقيقة)\)", row["reply"])
                self.assertEqual(sum(map(int, durations)), limit, row)
            if "horaire d’examen demain" in row["question"]:
                self.assertNotRegex(row["reply"], r"\b\d{1,2}\s*[:hH]\s*\d{2}\b")
            if "pour mon inscription" in row["question"]:
                self.assertNotRegex(
                    row["reply"].casefold(),
                    r"photocopie|(?:acte|extrait) de naissance|attestation de domicile|passeport|carte d’identité",
                )
