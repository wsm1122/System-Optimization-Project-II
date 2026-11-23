# Public experiment evidence

This directory contains a privacy-reduced subset of the R8-B and R9 experiment records.

## Files

- `r8b_process_summary_public.csv`: process-level aggregate rows for E1–E4 from run `combine_20261122T120238Z_cacaf324`.
- `r9_process_summary_public.csv`: process-level aggregate rows for E1/E2 from run `final_20261122T131820Z_93ef984b`.
- `summarize_public_runs.py`: recomputes worker-level summary statistics from those CSV files.
- `evidence_manifest.json`: provenance and field-redaction record.

The CSVs preserve the measured per-process aggregate metrics from the source `process_summary.csv` files, while omitting process IDs and, for R9, prediction hashes. They do not include per-utterance predictions, speaker IDs, per-audio timing rows, raw JSONL worker traces, audio, or host/GPU identifiers. These omissions are intentional; this is not the original raw evidence archive.

The reported per-audio paired confidence intervals cannot be independently recomputed from these process-level summaries alone. They should be checked against the original evidence supplied privately through the course's approved submission channel, if required. The original machine-specific worker logs must not be treated as public artifacts.

## Recompute

From this directory, run:

```bash
python summarize_public_runs.py
```

The script prints aggregate means and standard deviations across independent process rows. It does not rerun inference and does not reconstruct per-audio statistics.
