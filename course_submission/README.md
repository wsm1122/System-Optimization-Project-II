# AISHELL-1 kNN-CTC course submission materials

This directory collects the course report, AI-use statement, reproduction instructions, experiment code, frozen configuration, formal-verification materials, and compact R8-B/R9 result evidence from the local R10 submission package.

## Included

- Final report PDF, public-safe stage reports, README, reproduction prompt, and AI-assistance statement.
- The core optimization implementations and tests are in the repository root (`src/`, `tests/`, `configs/`). This folder adds the course-facing report, AI-use materials, and formal-verification code. It does not mirror every historical stage worker/orchestrator.
- Frozen configuration and asset/environment audit records.
- The root README records the aggregate R8-B/R9 outcomes. This public folder does not contain the raw per-audio result tables.
- Third-party inference support source and its license notice where included in the original package.

## Not included

- AISHELL-1 audio, model checkpoints, Datastore keys/values, FAISS indexes, virtual environments, or captured intermediate tensors.
- Dataset manifests/reference corpora, per-utterance predictions, speaker/audio-level timing tables, and repeated verbose worker traces/logs. Those are data-derived artifacts and are not published in this public mirror; the reports identify the source run IDs and the local R10 evidence package records their full SHA-256 inventory.
- Internal hostnames, process IDs, GPU UUIDs, and server-specific absolute paths are excluded from the public mirror.
- Some detailed Markdown diagnostics that contain those identifiers are also omitted; the final report PDF and public-safe reports provide the course-facing account.

Therefore this is a browsable course-submission mirror, **not a byte-for-byte copy of the 43 MiB local evidence ZIP** and not a stand-alone executable reproduction. Reproducing GPU results requires the public AISHELL-1 data, compatible model/index assets, and the documented GPU environment. No historical measurement is represented as a fresh run.

The original project root files remain available; this folder is additive and does not replace them.
