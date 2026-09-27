# Amazon ML Challenge 2026 — Methodology

## 1. Problem formulation

Business Entity Resolution across Source 1, Source 2 and Source 3.

## 2. Data understanding

Use measured dataset statistics from `reports/eda_report.md`. Bounded smoke
reports are diagnostic and must be kept separate from full-data metrics.

## 3. Normalization

Describe the deterministic name/address/country normalization rules used.

## 4. Candidate generation / blocking

The blocker builds reusable target indexes for exact normalized name and
address, country/name and country/address combinations, token postings, and
name/address prefixes. Each rule has a configurable posting limit and the
union has a configurable per-source candidate cap. Candidate recall is written
to `reports/candidate_recall.json`.

## 5. Feature engineering

Describe name, address, country and interaction features.

## 6. Hard negative mining

Describe how difficult non-matching pairs were selected without using external data.

## 7. Model architecture

Describe the final model, hyperparameters, class balancing and training procedure.

## 8. Threshold selection

Describe validation-based F0.5 threshold optimization and singleton/no-match handling.

## 9. Experiments

Summarize controlled experiments and validation metrics.

## 10. Error analysis

Describe false positives, false negatives and observed failure modes.

## 11. Final results

Use actual measured results only.

## 12. Reproducibility

Give exact environment and commands.

## 13. Compliance

State that the pipeline uses only challenge-provided data and does not perform external entity lookup.
