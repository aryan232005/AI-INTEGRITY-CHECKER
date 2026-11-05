# Batch summary

Documents: 15 (13 processed, 2 skipped) | embedding: **all-MiniLM-L6-v2** | match threshold 0.75

## Risk overview

| Document | Risk | Peer % | External % | AI % | Top overlap |
|---|---|---|---|---|---|
| 01_market_entry_vietnam.docx | HIGH | 100.0 | 0.0 | 57.2 | 02_vietnam_entry_strategy.docx (100.0%) |
| 02_vietnam_entry_strategy.docx | HIGH | 100.0 | 0.0 | 57.2 | 01_market_entry_vietnam.docx (100.0%) |
| 04_pitch_supply_chain.mp4 | HIGH | 100.0 | 0.0 | n/a | 03_supply_chain_report.pdf (100.0%) |
| 12_snack_market_partnering.docx | HIGH | 86.2 | 0.0 | 45.8 | 02_vietnam_entry_strategy.docx (86.2%) |
| 14_bakery_mill_report.pdf | HIGH | 0.0 | 52.7 | 11.6 | vertical_integration.txt [ref] (52.7%) |
| 03_supply_chain_report.pdf | MEDIUM | 30.3 | 0.0 | 51.1 | 04_pitch_supply_chain.mp4 (30.3%) |
| 05_ai_education_essay.docx | MEDIUM | 0.0 | 0.0 | 99.8 | - |
| 06_case_alpha.pdf | LOW | 0.0 | 0.0 | 3.8 | - |
| 07_case_bravo.docx | LOW | 0.0 | 0.0 | 9.0 | - |
| 08_case_charlie.pdf | LOW | 0.0 | 0.0 | 8.6 | - |
| 09_case_delta.docx | LOW | 0.0 | 0.0 | 13.3 | - |
| 13_original_pitch_bikes.mp4 | LOW | 0.0 | 0.0 | n/a | - |
| 15_urban_beekeeping.pdf | LOW | 0.0 | 0.0 | 7.5 | - |
| 10_corrupted_empty.pdf | UNPROCESSABLE | - | - | - | File is empty (0 bytes) |
| 11_silent_pitch.mp4 | UNPROCESSABLE | - | - | - | MP4 audio track is silent - nothing to transcribe |

## Peer overlap matrix (% of ROW document's words matched in COLUMN document)

| |01|02|03|04|05|06|07|08|09|12|13|14|15|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 01_market_entry_vietnam.docx |-|100|.|.|.|.|.|.|.|100|.|.|.|
| 02 02_vietnam_entry_strategy.docx |100|-|.|.|.|.|.|.|.|100|.|.|.|
| 03 03_supply_chain_report.pdf |.|.|-|30|.|.|.|.|.|.|.|.|.|
| 04 04_pitch_supply_chain.mp4 |.|.|100|-|.|.|.|.|.|.|.|.|.|
| 05 05_ai_education_essay.docx |.|.|.|.|-|.|.|.|.|.|.|.|.|
| 06 06_case_alpha.pdf |.|.|.|.|.|-|.|.|.|.|.|.|.|
| 07 07_case_bravo.docx |.|.|.|.|.|.|-|.|.|.|.|.|.|
| 08 08_case_charlie.pdf |.|.|.|.|.|.|.|-|.|.|.|.|.|
| 09 09_case_delta.docx |.|.|.|.|.|.|.|.|-|.|.|.|.|
| 12 12_snack_market_partnering.docx |86|86|.|.|.|.|.|.|.|-|.|.|.|
| 13 13_original_pitch_bikes.mp4 |.|.|.|.|.|.|.|.|.|.|-|.|.|
| 14 14_bakery_mill_report.pdf |.|.|.|.|.|.|.|.|.|.|.|-|.|
| 15 15_urban_beekeeping.pdf |.|.|.|.|.|.|.|.|.|.|.|.|-|

Timings (s): {'ingestion': 27.96, 'indexing': 67.13, 'scoring': 0.03, 'total': 96.03}