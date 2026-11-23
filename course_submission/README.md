# AISHELL-1 kNN-CTC course submission materials

This directory collects the course report, AI-use statement, reproduction instructions, selected stage reports, frozen configuration, and formal-verification materials from the local R10 submission package.

## Included

- Final report PDF, public-safe stage reports, README, reproduction prompt, and AI-assistance statement.
- The core optimization implementations and tests are in the repository root (`src/`, `tests/`, `configs/`). This folder adds the course-facing report, AI-use materials, and formal-verification code. It does not mirror every historical stage worker/orchestrator.
- Frozen configuration and asset/environment audit records.
- Privacy-reduced process-level R8-B/R9 summaries and a summary-recalculation script are in `evidence/`. The summaries retain independent worker rows but omit process IDs and prediction hashes.
- The Z3 verification script, dependency pin, and solver result are under `formal_verification/`.

## Not included

- AISHELL-1 audio, model checkpoints, Datastore keys/values, FAISS indexes, virtual environments, or captured intermediate tensors.
- Dataset manifests/reference corpora, per-utterance predictions, speaker/audio-level timing tables, and repeated verbose worker traces/logs. These are not published in this public mirror. The evidence README explains the scope and limitations of the public summaries.
- Internal hostnames, process IDs, GPU UUIDs, and server-specific absolute paths are excluded from the public mirror.
- Some detailed Markdown diagnostics that contain those identifiers are also omitted; the final report PDF and public-safe reports provide the course-facing account.

The public process summaries support checking worker-level aggregate metrics; they do **not** reproduce per-audio paired confidence intervals or provide per-utterance prediction evidence. Those require the original evidence through a course-approved private submission channel, if requested.

Therefore this is a browsable course-submission mirror, **not a byte-for-byte copy of the 43 MiB local evidence ZIP** and not a stand-alone executable reproduction. Reproducing GPU results requires the public AISHELL-1 data, compatible model/index assets, and the documented GPU environment. No historical measurement is represented as a fresh run.

The original project root files remain available; this folder is additive and does not replace them.
