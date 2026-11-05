# Evaluation

## What was run, and what was not

**Main results (real models).** Run on the author's Windows machine (CPU only) with
`all-MiniLM-L6-v2` for embeddings and `faster-whisper` **base** (int8) for speech-to-text.
The two spoken MP4s were transcribed by Whisper for real: the sidecar transcript files were
hidden during the run to force this (see failure case 1 for why that mattered). AI-likelihood
uses the built-in stylometric heuristic.

**Ablation (fallback embedder).** An earlier run used the offline TF-IDF fallback because
model downloads were blocked in my first environment. I keep those numbers as a comparison,
because they show *why* a neural embedder is needed.

**Not verified:** the optional RoBERTa AI detector (`--ai_backend roberta`), any real student
documents or human recordings, non-English text, and anything at the 10,000-file scale
(the scaling section of the README is design reasoning, not measurement).

## Dataset and ground truth

15 synthetic files (6 PDF, 6 DOCX, 3 MP4) in `sample_submissions/`, ground truth in
`dataset_manifest.json`. All text was written for this project and nothing is real student work.
The "original" texts, the paraphrase and the "LLM-generated" essay were all written by an LLM
(the coding assistant), so the data is easier and cleaner than real submissions.

| Relationship | Files |
|---|---|
| Exact copy | 01 and 02 (DOCX) |
| Paraphrase (synonym rewording) | 12 vs 01 and 02 |
| Cross-modal | 03 (PDF, page 2) read aloud in 04 (MP4) |
| External source copy | 14 contains a paragraph copied from `knowledge_base/vertical_integration.txt` |
| AI-generated | 05 |
| Shared 150-word prompt, original answers | 06-09 |
| Independent originals | 13, 15 (and the answers in 06-09) |
| Degraded | 10 (0-byte PDF), 11 (silent MP4) |

## Results (MiniLM + Whisper base)

**Pair-level overlap detection (78 document pairs, flag if >= 15% overlap)**

| | Predicted flagged | Predicted clean |
|---|---|---|
| **Actually positive** | TP = 4 | FN = 0 |
| **Actually negative** | FP = 0 | TN = 74 |

Precision = 1.00, Recall = 1.00, F1 = 1.00. Recall by type: exact copy 1/1, paraphrase 2/2, cross-modal 1/1.

**External-source plagiarism (13 documents, flag >= 15%)**

| | Predicted flagged | Predicted clean |
|---|---|---|
| **Actually positive** | TP = 1 | FN = 0 |
| **Actually negative** | FP = 0 | TN = 12 |

Precision = 1.00, Recall = 1.00, F1 = 1.00.

**AI-generated detection (11 text documents, flag >= 70%)**

| | Predicted flagged | Predicted clean |
|---|---|---|
| **Actually positive** | TP = 1 | FN = 0 |
| **Actually negative** | FP = 0 | TN = 10 |

Precision = 1.00, Recall = 1.00, F1 = 1.00.

**Shared prompt:** 0 of 6 template-sharing pairs (06-09) falsely flagged; the 5 shared sentences
were excluded and each of the four case answers scored 0% peer overlap and LOW risk.

**Robustness:** the two unreadable files (10, 11) were skipped with a warning, exactly matching
the manifest; the other 13 files were processed.

**Best passage-cosine per document pair**

| Group | n | min | mean | max |
|---|---|---|---|---|
| True overlap pairs | 4 | 0.879 | 0.921 | 1.000 |
| Unrelated pairs | 74 | 0.079 | 0.374 | 0.684 |

The match threshold (0.75, from the brief) sits inside the gap between the highest unrelated
pair (0.684) and the lowest true pair (0.879). It was not tuned on this data. I have not
checked which unrelated pair scored 0.684.

Speed: 15 files end to end in about 96 s on a CPU-only laptop (about 27 s ingestion including
Whisper, about 67 s indexing, mostly model loading and embedding). Scoring itself takes under a second.

### Ablation: what the neural embedder buys

| Embedder | Pairs found | Paraphrase pairs found | Lowest true-pair cosine | Highest unrelated cosine |
|---|---|---|---|---|
| TF-IDF 1-2-grams (offline fallback) | 2 of 4 | 0 of 2 | 0.205 | 0.534 |
| all-MiniLM-L6-v2 | 4 of 4 | 2 of 2 | 0.879 | 0.684 |

With TF-IDF the paraphrase pairs scored *below* the highest unrelated pair, so no threshold could
have separated them. The representation was the problem, not the cut-off. (TF-IDF row measured
in my first environment with sidecar transcripts for the videos.)

## Scenario checklist

| # | Scenario | Result | Notes |
|---|---|---|---|
| 1 | Cross-modal duplication | **Pass** | 04 matched to 03 page 2 with timestamps `00:00:00 - 00:00:25`, cosine 0.997 and 0.871, using real Whisper output that contained recognition errors. |
| 2 | Paraphrased content | **Pass with MiniLM**, fail with TF-IDF | 12 flagged at 86.2% overlap. Only synonym-style rewording was tested. |
| 3 | Corrupted / empty | **Pass** | 0-byte PDF and silent MP4 produce `UNPROCESSABLE` reports; the batch finishes. Unit tests also cover garbage PDF/DOCX/MP4 bytes and an MP4 with no audio stream. |
| 4 | Boilerplate prompt | **Pass** | See above. |
| 5 | Multi-document batch | **Pass** | 15 files, one JSON + one Markdown report each, plus `summary.md` and the overlap matrix. |

### Reading the tables

Only 4 true overlap pairs and 1 positive each for external and AI detection exist, so a single
miss would move recall by 25-100 points. Treat these as smoke-test evidence that the pipeline works
on clear cases, not as estimates of real-world accuracy. Perfect scores on synthetic data written by
the same assistant that wrote the detector's features are the weakest kind of evidence.

## Failure analysis and issues found

**1. Whisper was failing silently behind the fallback (found and fixed).** On the first real-model
run the video results looked fine, but the word counts (58 and 56) were identical to the sidecar
transcripts, and the report warnings said "Whisper unavailable (TypeError)". Hiding the sidecar files
turned both videos into `UNPROCESSABLE`, which proved Whisper had never run. A debug script
(`scripts/debug_whisper.py`) located the cause: `faster-whisper` 1.2.1 calls
`av.open(..., metadata_errors=...)`, which PyAV 19.0.2 no longer accepts. *Fix:* we already convert
audio to a 16 kHz WAV with ffmpeg, so the extractor now loads the samples itself and passes the array
to Whisper, bypassing that decoder. *Lesson:* a fallback that hides a failure is dangerous in an
evaluation. The sidecar path stays, but every report and `summary.md` now says when it was used,
the warning includes the full error text, and the tests force it deliberately.

**2. Speech-recognition noise, and a misleading `DIRECT_COPY` label.** Whisper-base is imperfect on
the synthetic voice. In one run it produced "Roger Dam Herb" for "Rotterdam hub" and "Too long haul
carriers" for "Two long-haul carriers"; in the pipeline run the same words came out closer to correct
("Rotterdam Hub", but still "Too long haul carriers ... a breed delivery window"). So transcripts of the
same file differed between runs (cause not investigated; possibly Whisper's temperature fallback on
low-confidence audio). MiniLM still matched the passages (cosine 0.997 and 0.871), but the second match has a
word-overlap Jaccard of only 0.246 and is still labelled `DIRECT_COPY`, because the label is decided by
cosine alone. A reviewer reading only the label would think the words were copied verbatim. *Mitigation
today:* the report prints Jaccard beside cosine. *Better fix (not done):* require both high cosine and
high Jaccard for `DIRECT_COPY`, and call anything else a semantic match when the source is a transcript.
Also untested: noisier real recordings, accents, or a larger Whisper model.

**3. The copied-from file is also flagged, and direction is ambiguous.** 03 is the PDF that was read
aloud, yet it gets MEDIUM (30.3% overlap) because similarity is symmetric. The engine now adds a
`direction_hints` entry using containment asymmetry: 100% of 04 appears in 03, but only 30% of 03 appears
in 04, so 04 is the derived file. That is only a hint: it also fires when a short text is quoted inside a
long one, and it says nothing about who submitted first (the files carry no timestamps). Pairs of equal
size, such as the exact copies 01 and 02, get no direction at all.

**4. AI-likelihood is fragile and weakly validated.** The heuristic scored the unremarkable formal
documents 01/02 at 57.2, 03 at 51.1 and 12 at 45.8, against 99.8 for the LLM-style essay. The 70% flag
separates them here, but there is one positive example, the essay contains the stock phrases the heuristic
looks for, and the "human" documents were also written by an LLM. Formal prose, non-native writers and
lightly edited LLM text will cause errors in both directions. The feature weights are hand-chosen and not
fitted to data (stated in the code). AI-only cases are capped at MEDIUM so this signal can never alone
produce a HIGH verdict.

**5. The paraphrase test is easy.** File 12 swaps synonyms and re-orders clauses but keeps the same
structure and facts. Harder paraphrases (restructured arguments, summaries, translation-and-back,
LLM rewrites with new examples) are untested, and MiniLM is known to be weaker on those. Borderline
pairs are candidates for cross-encoder re-ranking.

**Known limitations (reasoned, not measured).** Coverage marks every sentence inside a matching window, so
partial overlap can be slightly over-counted. The relaxed boilerplate rule (3+ documents, head or tail
position) could excuse a small group copying a block at the start of their documents, which is why the
excluded sentences are listed in each report. Image-only PDFs are rejected, not OCR'd. Documents under 80
words get no AI score. English only.

## Trade-off analysis

- **Embedding model.** `all-MiniLM-L6-v2` (22M parameters, 384 dimensions, ~90 MB) runs on a laptop CPU and
  separated every pair here with a clear margin. Larger models (e.g. `all-mpnet-base-v2`, BGE or E5 base)
  typically help on harder paraphrase at several times the cost; I would benchmark one on real data before
  switching. The TF-IDF fallback exists so the pipeline still runs offline, but the ablation shows it is not a
  substitute.
- **Whisper size.** `base` is fast but made visible errors on this voice. `small` would likely be more
  accurate at a few times the runtime; untested here. Since matching is semantic, small errors cost little, but
  they hurt the lexical evidence shown to the reviewer.
- **Vector index.** `IndexFlatIP` is exact with no parameters, so any miss is the model's fault and never the
  index's. HNSW only pays off at a much larger scale (see README).
- **Passages vs chunks.** 200-300-word chunks would dilute a 50-word spoken pitch and blur the evidence.
  Passage windows give exact, quotable evidence and sentence-level coverage; chunks remain as a coarse
  section-level view.
- **Coverage vs averaged cosine.** Coverage answers "how much of this submission is borrowed", which is what a
  reviewer needs; an average would shrink when a copied paragraph sits inside a long original.

## Reproducing

```
# Windows (PowerShell)                         # Mac / Linux
python -m venv .venv                           bash setup.sh
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py
python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports --knowledge_base ./knowledge_base
python scripts/evaluate.py
python -m unittest discover -s tests -v       # 18 tests; Whisper is disabled inside the tests
python scripts/debug_whisper.py               # only if video transcription fails
```