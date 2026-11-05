# Integrity report - `03_supply_chain_report.pdf`

**Risk: MEDIUM** | type: pdf | words analysed: 271

## Scores

| Signal | Value |
|---|---|
| Internal peer similarity | 30.3% |
| External (reference) match | 0.0% |
| AI-generated likelihood | 51.1% |

## Why

30.3% of the original content overlaps with other submissions (04_pitch_supply_chain.mp4: 30.3%). Some matches are paraphrases: high semantic similarity but low word-for-word overlap (Jaccard as low as 0.343). Cross-modal matches present (video transcript <-> document); see multimodal_matches for timestamps and pages. Direction of copying cannot be determined from content alone - check submission times.

## Overlap by document

| Matched file | % of this document covered |
|---|---|
| 04_pitch_supply_chain.mp4 | 30.3% |

## Cross-modal matches (video <-> document)

**1. 04_pitch_supply_chain.mp4** - cosine **0.998** (DIRECT_COPY, word-overlap Jaccard 0.839)

- This document: page 2
- Matched document: time 00:00:08 - 00:00:19

> **Submitted:** Fuel surcharges rose faster than we could pass them on to customers. Two long-haul carriers failed to meet agreed delivery windows, and we paid penalty fees on thirty-one shipments.
>
> **Source:** Fuel surcharges rose faster than we could pass on to customers. Two long-haul carriers failed to meet agreed delivery windows, and we paid penalty fees on thirty-one shipments.

**2. 04_pitch_supply_chain.mp4** - cosine **0.773** (PARAPHRASE, word-overlap Jaccard 0.343)

- This document: page 2
- Matched document: time 00:00:20 - 00:00:25

> **Submitted:** In response we have moved forty percent of Rotterdam volume to the Antwerp terminal. We also renegotiated the carrier contracts to include stricter penalties. Early data from October suggests transit times have improved by almost a day.
>
> **Source:** In response we have moved forty percent of Rotterdam volume to the Antwerp terminal.

---
_Embedding: all-MiniLM-L6-v2 | thresholds: match >= 0.75, direct >= 0.85 | AI backend: stylometric-heuristic_