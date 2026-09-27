# Amazon ML Challenge 2026 - Business Entity Resolution

This project contains the current scalable candidate-generation foundation for
the Amazon ML Challenge business entity-resolution task.

The task is multi-match entity resolution: one Source 1 record may match zero,
one, or many Source 2 and Source 3 records. The implementation does not assume
one-to-one matching.

## Current Status

Completed:

- Inspected the existing data-loading, normalization, blocking, feature, and
  evaluation code.
- Replaced brute-force TF-IDF nearest-neighbor blocking with reusable inverted
  indexes.
- Added exact, country-aware, token, and prefix blocking rules.
- Added configurable candidate budgets and per-rule posting limits.
- Added compact/token normalization representations.
- Added explicit missing-address features.
- Added deterministic stratified candidate-recall benchmarking.
- Measured recall using the actual full training target sources and
  `train_ground_truth.tsv`.

Not yet completed:

- Blocking improvements needed to raise candidate recall.
- Hard-negative mining.
- LightGBM training and threshold tuning.
- Full test inference.
- Official submission validation and ZIP creation.

No training, full inference, or final submission files have been created yet.

## Data Location

The supplied data is expected at:

```text
D:\rig\mlchallenge\student_resource
```

The project discovers these files automatically:

```text
student_resource/
  dataset/
    train/
      train_source1.tsv
      train_source2.tsv
      train_source3.tsv
      train_ground_truth.tsv
    test/
      test_source1.tsv
      test_source2.tsv
      test_source3.tsv
```

Override the location with `--data-root` when necessary.

## Setup

Python 3.10 or newer is recommended.

```powershell
Set-Location D:\rig\mlchallenge\amazon_ml_challenge_project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Training now requests the CUDA tree learner by default. The standard PyPI
LightGBM wheel is CPU-only, so GPU training requires a LightGBM build compiled
with `-DUSE_CUDA=1` and a working CUDA toolkit. Verify the build before a full
run with:

```powershell
python -c "import lightgbm as lgb, numpy as np; lgb.LGBMClassifier(n_estimators=2, device_type='cuda').fit(np.array([[0.],[1.],[2.],[3.]]), np.array([0,0,1,1]))"
```

Run on the first GPU with the default settings:

```powershell
python -m src.pipeline --data-root D:\rig\mlchallenge\student_resource --mode all
```

Use `--gpu-device-id 1` for another GPU. Use `--device cpu` only when a
CUDA-enabled LightGBM build is unavailable. Candidate generation, text
normalization, and feature construction remain CPU operations; LightGBM model
training uses the selected device.

The implementation uses only the supplied challenge data. It does not call
business registries, web search, geocoding services, external APIs, or outside
business databases.

## Implemented Blocking

`src/blocking.py` provides `TargetIndex`, which is built once per target source
and reused for candidate generation. It does not create a target-wide TF-IDF
matrix and does not run brute-force nearest-neighbor search.

The current blocking rules are:

- `exact_name`
- `exact_address`
- `country_name`
- `country_address`
- `name_token`
- `address_token`
- `prefix_name`
- `prefix_address`

Candidate provenance is retained when requested. Candidate limits are exposed
through `src/config.py`, including the overall per-source budget, token posting
limit, prefix posting limit, prefix length, and token limits.

## Candidate-Recall Benchmark

The benchmark samples Source 1 records from the full ground truth by match-count
bucket. The initial sample used 5,000 rows:

- 500 zero-match records
- 500 one-match records
- 500 two-match records
- 500 three-match records
- 500 four-match records
- 2,500 five-plus-match records

Zero-match records are excluded from recall calculations and reported separately
as singleton behavior.

Run the benchmark:

```powershell
python -m src.pipeline `
  --data-root D:\rig\mlchallenge\student_resource `
  --mode benchmark `
  --sample-size 5000 `
  --random-state 42
```

The default benchmark evaluates per-source budgets of 50, 80, 120, and 160.
To evaluate a custom budget, repeat the option:

```powershell
python -m src.pipeline --mode benchmark --sample-size 5000 `
  --candidate-budget 80 --candidate-budget 160
```

The benchmark writes:

- `reports/candidate_recall_benchmark.json`
- `reports/candidate_recall_benchmark.md`
- `reports/missed_candidates_sample.tsv` with at most 500 missed pairs

### Measured Benchmark Results

These results were measured with random seed 42 and the full Source 2/Source 3
training target files:

| Per-source budget | Average candidates per S1 | Overall recall | S2 recall | S3 recall |
|---:|---:|---:|---:|---:|
| 50 | 100.00 | 42.95% | 44.86% | 41.14% |
| 80 | 160.00 | 51.06% | 54.25% | 48.06% |
| 120 | 240.00 | 62.57% | 65.64% | 59.69% |
| 160 | 319.65 | 63.62% | 66.70% | 60.71% |

The combined average is approximately twice the configured budget because the
budget is applied independently to Source 2 and Source 3.

The benchmark shows that Source 3 is consistently harder and that increasing
the budget from 120 to 160 gives only a small recall improvement. The current
recall is not yet strong enough to justify model training, so blocking should be
improved before proceeding.

## Other CLI Modes

Generate an EDA report:

```powershell
python -m src.pipeline --data-root D:\rig\mlchallenge\student_resource --mode eda
```

The older bounded recall mode is available for smoke testing, but it truncates
target files and is not a valid quality benchmark:

```powershell
python -m src.pipeline --data-root D:\rig\mlchallenge\student_resource `
  --mode candidate-recall --sample-s1 1000
```

Do not interpret metrics from that mode as final candidate recall.

Training and inference modes exist in the current pipeline but have deliberately
not been run on the full dataset:

```powershell
python -m src.pipeline --data-root D:\rig\mlchallenge\student_resource --mode train
python -m src.pipeline --data-root D:\rig\mlchallenge\student_resource --mode inference
```

## Reports and Documentation

- [reports/candidate_recall_benchmark.json](reports/candidate_recall_benchmark.json)
- [reports/candidate_recall_benchmark.md](reports/candidate_recall_benchmark.md)
- [reports/missed_candidates_sample.tsv](reports/missed_candidates_sample.tsv)
- [Documentation_template.md](Documentation_template.md)

## Next Recommended Work

1. Analyze missed pairs using the generated missed-candidate sample.
2. Add compact-name, significant-token, address-number, and postal-code blocks
   where supported by the supplied fields.
3. Re-run the same stratified benchmark and compare recall by source and bucket.
4. Only after candidate recall is acceptable, implement hard-negative mining and
   begin bounded pair-feature/model smoke tests.
5. Keep full inference and official validation deferred until the model and
   threshold are validated.