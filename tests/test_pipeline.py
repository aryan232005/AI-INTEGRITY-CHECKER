"""Run with:  python -m unittest discover -s tests -v"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.boilerplate import find_boilerplate  # noqa: E402
from src.config import Config  # noqa: E402
from src.embedder import TfidfEmbedder  # noqa: E402
from src.extractor import extract_file, fmt_range, normalize_text  # noqa: E402
from src.indexer import Corpus, Sentence, build_chunks, split_sentences  # noqa: E402
from src.detector import IntegrityEngine  # noqa: E402

SAMPLES = os.path.join(ROOT, "sample_submissions")


def extract_offline(path):
    """Extract a file with Whisper disabled so tests stay fast and deterministic.

    Videos then use their '<name>.transcript.json' sidecar fixtures.  Real
    Whisper is exercised by running the pipeline itself (see EVALUATION.md)."""
    with mock.patch("src.extractor._transcribe_whisper",
                    side_effect=RuntimeError("tests run offline")):
        return extract_file(path)


class TestRobustness(unittest.TestCase):
    def test_zero_byte_pdf(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "a.pdf")
            with open(p, "wb"):
                pass
            d = extract_file(p)
            self.assertFalse(d.ok)
            self.assertIn("empty", d.error)

    def test_garbage_files_do_not_raise(self):
        with tempfile.TemporaryDirectory() as t:
            for name in ("x.pdf", "x.docx", "x.mp4"):
                p = os.path.join(t, name)
                with open(p, "wb") as fh:
                    fh.write(b"this is not a real file" * 10)
                d = extract_file(p)
                self.assertFalse(d.ok, name)
                self.assertTrue(d.warnings)

    def test_unsupported_and_missing(self):
        self.assertFalse(extract_file("/nonexistent/file.pdf").ok)
        self.assertFalse(extract_file(os.path.join(ROOT, "README.md")).ok)

    def test_mp4_without_audio_track(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "v.mp4")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                            "color=c=black:s=160x90:r=5", "-t", "1", "-an", p], check=True)
            d = extract_file(p)
            self.assertFalse(d.ok)
            self.assertIn("no audio track", d.error)

    def test_silent_mp4(self):
        d = extract_file(os.path.join(SAMPLES, "11_silent_pitch.mp4"))
        self.assertFalse(d.ok)
        self.assertIn("silent", d.error)


class TestExtraction(unittest.TestCase):
    def test_pdf_pages_and_header_footer_removed(self):
        d = extract_file(os.path.join(SAMPLES, "03_supply_chain_report.pdf"))
        self.assertTrue(d.ok)
        self.assertEqual([s.page for s in d.segments], [1, 2, 3])
        blob = " ".join(s.text for s in d.segments)
        self.assertNotIn("Confidential - Page", blob)
        self.assertNotIn("Quarterly Operations Review", blob)

    def test_video_timestamps(self):
        d = extract_offline(os.path.join(SAMPLES, "04_pitch_supply_chain.mp4"))
        self.assertTrue(d.ok)
        self.assertTrue(all(s.start is not None and s.end > s.start for s in d.segments))
        self.assertEqual(fmt_range(24, 36), "00:00:24 - 00:00:36")

    def test_normalize(self):
        self.assertEqual(normalize_text("inte-\ngration   of\n\n\n\nsuppliers"),
                         "integration of\n\nsuppliers")


class TestIndexing(unittest.TestCase):
    def test_sentence_split_keeps_abbreviations_and_decimals(self):
        s = split_sentences("Margins fell 4.2% in Q3, e.g. in Rotterdam. Costs rose. Done!")
        self.assertEqual(len(s), 3)

    def test_chunk_size_and_overlap(self):
        sents = [Sentence("d", i, " ".join(["w%d" % i] * 20)) for i in range(60)]  # 1200 words
        chunks = build_chunks("d", "docx", sents, 250, 50)
        self.assertGreater(len(chunks), 3)
        for c in chunks[:-1]:
            self.assertTrue(200 <= c.n_words <= 300, c.n_words)
        self.assertLessEqual(chunks[1].s_from, chunks[0].s_to)   # overlapping windows

    def test_boilerplate_df_rule(self):
        cfg = Config()
        tmpl = "Please read the case prompt carefully before you answer anything."
        sm = {f"d{i}": [Sentence(f"d{i}", 0, tmpl),
                        Sentence(f"d{i}", 1, " ".join(f"original{i}analysis{k}"
                                                      for k in range(60)))]
              for i in range(4)}
        flagged, info = find_boilerplate(sm, cfg)
        self.assertTrue(all(0 in flagged[d] and 1 not in flagged[d] for d in sm))
        # a verbatim sentence shared by only two documents is NOT boilerplate
        sm2 = {f"d{i}": [Sentence(f"d{i}", 0, tmpl if i < 2 else f"other text number {i} here ok")]
               for i in range(6)}
        flagged2, _ = find_boilerplate(sm2, cfg)
        self.assertFalse(any(flagged2[d] for d in sm2))


class TestBoilerplateEdgeCases(unittest.TestCase):
    TEXT = "The quarterly review shows that margins fell because fuel costs rose sharply again."

    def _docs(self, texts):
        return {f"d{i}": [Sentence(f"d{i}", j, t) for j, t in enumerate(ts)]
                for i, ts in enumerate(texts)}

    def test_exact_copy_pair_in_small_batch_is_not_boilerplate(self):
        other = ["Completely different content about beekeeping on city rooftops in summer."]
        flagged, _ = find_boilerplate(self._docs([[self.TEXT], [self.TEXT], other]), Config())
        self.assertFalse(any(flagged.values()))

    def test_identical_trio_has_no_original_content_so_is_still_scored(self):
        flagged, _ = find_boilerplate(self._docs([[self.TEXT]] * 3), Config())
        self.assertFalse(any(flagged.values()))

    def test_template_with_original_answers_is_still_excluded(self):
        def own(i):
            return " ".join(f"unique{i}word{k}" for k in range(60))

        docs = self._docs([[self.TEXT, own(i)] for i in range(4)])
        flagged, _ = find_boilerplate(docs, Config())
        self.assertTrue(all(flagged[d] == {0} for d in docs))


class TestCli(unittest.TestCase):
    def test_all_files_unreadable_does_not_crash(self):
        import run_integrity_check
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as out:
            with open(os.path.join(src, "a.pdf"), "wb") as fh:
                fh.write(b"junk")
            open(os.path.join(src, "b.docx"), "wb").close()
            code = run_integrity_check.main(["--input_dir", src, "--output_dir", out,
                                             "--embedder", "tfidf"])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(out, "a.pdf.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "summary.md")))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = sorted(f for f in os.listdir(SAMPLES) if f.endswith((".pdf", ".docx", ".mp4")))
        docs = [extract_offline(os.path.join(SAMPLES, n)) for n in names]
        cfg = Config()
        cls.corpus = Corpus(TfidfEmbedder(), cfg)       # deterministic, offline
        cls.corpus.prepare([d for d in docs if d.ok])
        cls.corpus.build([])
        cls.engine = IntegrityEngine(cls.corpus, cfg)

    def test_no_self_matches(self):
        for doc_id in self.corpus.docs:
            r = self.engine.analyse(doc_id)
            for m in r.peer_matches + r.multimodal_matches:
                self.assertNotEqual(m["matched_file"], doc_id)

    def test_exact_copy_flagged(self):
        r = self.engine.analyse("01_market_entry_vietnam.docx")
        self.assertGreaterEqual(r.scores["peer_similarity_percentage"], 90)
        self.assertEqual(r.peer_matches[0]["matched_file"], "02_vietnam_entry_strategy.docx")

    def test_cross_modal_has_timestamp_and_page(self):
        r = self.engine.analyse("04_pitch_supply_chain.mp4")
        loc = r.multimodal_matches[0]["location"]
        self.assertIn("query_timestamp", loc)
        self.assertEqual(loc["matched_page"], 2)

    def test_boilerplate_not_plagiarism(self):
        for n in ("06_case_alpha.pdf", "07_case_bravo.docx", "08_case_charlie.pdf",
                  "09_case_delta.docx"):
            r = self.engine.analyse(n)
            self.assertEqual(r.scores["peer_similarity_percentage"], 0.0, n)
            self.assertTrue(r.boilerplate)
            self.assertEqual(r.risk, "LOW")

    def test_independent_docs_low(self):
        self.assertEqual(self.engine.analyse("15_urban_beekeeping.pdf").risk, "LOW")

    def test_external_match(self):
        from src.extractor import extract_text_file
        kb = os.path.join(ROOT, "knowledge_base")
        refs = [extract_text_file(os.path.join(kb, f)) for f in sorted(os.listdir(kb))
                if f.endswith(".txt")]
        names = sorted(f for f in os.listdir(SAMPLES) if f.endswith((".pdf", ".docx", ".mp4")))
        docs = [d for d in (extract_offline(os.path.join(SAMPLES, n)) for n in names) if d.ok]
        cfg = Config()
        corpus = Corpus(TfidfEmbedder(), cfg)
        corpus.prepare(docs)
        corpus.build(refs)
        engine = IntegrityEngine(corpus, cfg)
        copied = engine.analyse("14_bakery_mill_report.pdf")
        self.assertGreaterEqual(copied.scores["external_similarity_percentage"], 40)
        self.assertEqual(copied.external_matches[0]["matched_file"], "vertical_integration.txt")
        self.assertEqual(engine.analyse("15_urban_beekeeping.pdf")
                         .scores["external_similarity_percentage"], 0.0)

    def test_direction_hint(self):
        from src.detector import add_direction_hints
        results = [self.engine.analyse(n) for n in self.corpus.docs]
        add_direction_hints(results)
        video = next(r for r in results if r.doc_id == "04_pitch_supply_chain.mp4")
        self.assertTrue(any("03_supply_chain_report.pdf" in h for h in video.direction_hints))
        pdf = next(r for r in results if r.doc_id == "03_supply_chain_report.pdf")
        self.assertEqual(pdf.direction_hints, [])


if __name__ == "__main__":
    unittest.main()
