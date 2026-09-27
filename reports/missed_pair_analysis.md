# Missed Candidate Pair Analysis

This analysis uses the 500-row sample in `missed_candidates_sample.tsv`. The
sample contains missed true pairs from the existing benchmark output, not
synthetic examples.

## Observed Counts

| Observation | Count |
|---|---:|
| Missed rows analyzed | 500 |
| S2 pairs | 227 |
| S3 pairs | 273 |
| `prefix_or_combination_gap` diagnosis | 347 |
| `no_shared_name_token` diagnosis | 136 |
| `missing_address` diagnosis | 17 |
| Shared normalized name token | 361 |
| Shared normalized address token | 482 |
| Shared name and address token | 344 |
| Shared address but no shared name token | 138 |
| Shared numeric address component | 415 |
| Both rows contain numeric address components | 446 |
| Exact compact-name equality | 12 |
| Exact compact-address equality | 14 |
| Shared 5- or 6-digit postal-like code | 0 |

The existing diagnosis is therefore directionally useful, but it misses the
main address signal: most missed pairs share address tokens, while numeric
components provide a stronger order-independent key for reordered and noisy
addresses.

## Supported Patterns

### Compact and separator variation

The sample includes punctuation, spacing, and separator changes such as
`All Industrial Holdings` versus `All Industrial Inc`, `Ag-572` versus
`Ag-4-572`, and `Okeefe` records with suffix or ordering changes. Exact compact
equality is relatively uncommon, so compact keys are useful as a high-precision
route but cannot be the only recovery mechanism.

### Significant token and legal-suffix variation

Business-specific tokens recur while legal or structural tokens move or change:
`Private Limited`, `LLC`, `Inc`, `Services`, and `Company` appear in different
orders or on only one side. Examples include `Southern It Private Limited`
versus `Southern It Limited Services`, and `United Ministries II` versus
`II MINISTRIES UNITED`. The blocker should preserve informative tokens and cap
their posting lists rather than rely on the complete name.

### Address-number and address-token overlap

Numeric components overlap in 415 pairs. The requested example
`1824 Cabin Creek Road` versus `#1824 CABIN CREEK RD` is representative. Other
observed changes include leading zeros, letter suffixes, house-number prefixes,
street-type abbreviations, reordered components, and extra locality text.
Number-plus-significant-address-token blocks are therefore supported by the
sample and are kept bounded.

### Unicode and encoding noise

The sample contains accents and visually corrupted text, for example `Aeon`
versus `Áeon`, `Research` versus `Résearch`, and OCR-like substitutions such as
`Solutions` versus `5olutions`. Accent-folded compact keys help the safe
formatting cases; arbitrary character substitutions are not treated as exact
blocking evidence.

### Country-aware relationships

The missed sample contains both US and India records, and the same business or
address token can be common within a country. New token, compact, and numeric
keys remain country-qualified where the key is not already exact.

## Unsupported or Not Added

No pair in this sample shares a reliable 5- or 6-digit postal-like code. Postal
formats vary across the supplied countries and this sample does not justify a
country-independent extractor, so no postal-code block was added.

The sample also contains records with no usable address or highly unrelated
names. These are not addressed with broad fuzzy or Cartesian blocking because
that would create large false-positive posting lists.

## Regression Examples

The blocker smoke test verifies recovery of the four requested examples:

- `All Industrial Holdings` / `All Industrial Inc` with house number `1824`.
- `Services Aeon Technologies Private Limited` / accented and hyphenated
  `Services Áeon Technologies-Private Limited`.
- `Okeefe Artificial` / `Okeefe Services #84539` with shared address number
  `2833`.
- `Ag-572 Shalimar Bagh` / `Ag-4-572 Shalimar Bagh`.

These are regression checks for blocking routes, not a claim that every similar
pair is a true match.# Missed Candidate Analysis

This analysis uses the 500-row sample in `reports/missed_candidates_sample.tsv`.
The sample was produced from the stratified 5,000-S1 benchmark and is a bounded
inspection sample, not a complete census of all missed pairs.

## Existing Diagnosis Counts

| Diagnosis | Count |
|---|---:|
| prefix_or_combination_gap | 347 |
| no_shared_name_token | 136 |
| missing_address | 17 |

The missed sample contains 227 Source 2 pairs and 273 Source 3 pairs. Source 3
is over-represented in this sample, consistent with its lower benchmark recall.

## Observed Field Patterns

The following measurements compare normalized, punctuation-insensitive field
components within the 500 sampled missed pairs:

| Pattern | Pairs | Percentage |
|---|---:|---:|
| Shared significant name token | 361 | 72.2% |
| Shared address token | 482 | 96.4% |
| Shared numeric address component | 415 | 83.0% |
| Shared 4+ digit numeric component | 152 | 30.4% |
| Equal compact name after case/diacritic/separator folding | 43 | 8.6% |
| Same first four compact-name characters | 336 | 67.2% |

The 4+ digit metric is not treated as a postal-code measurement. In this data,
those values are also commonly house numbers, unit numbers, or other address
numbers, and there is no reliable country-independent postal field. No postal
blocking rule was added.

## Representative Examples

### Address-number variation

`All Industrial Holdings` versus `All Industrial Inc` shares the address number
`1824`:

- Source 1: `Dry Branch, 1824 Cabin Creek Road, WV`
- Target: `#1824 CABIN CREEK RD, DRY BRANCH, WV`

The new `address_number` and `address_number_token` blocks are intended to
recover this pattern without requiring complete address equality.

### Compact and accented name variation

`Services Aeon Technologies Private Limited` versus
`Services Áeon Technologies-Private Limited` demonstrates safe separator,
case, and diacritic normalization. Compact-name equality and compact-name
prefix blocks cover this pattern without removing meaningful business tokens.

### Shared address number with different names

`Okeefe Artificial` versus `Okeefe Services` shares address number `2833`:

- Source 1: `2833 Bishop Meade Road`
- Target: `2833 Bishop Meade Rd, PO Box 164, Boyce Township, Virginia`

The address-number family provides an independent route when names are noisy
or only partially overlapping.

### Name-token noise

Observed examples include spelling noise (`RK Great Advisors` versus
`RK 6reat Advisors`), transliteration or encoding variation, legal suffix
reordering, and names such as `Quality Asset Solutions, LLC` versus
`@qualityasset`. Significant-token blocks preserve informative business tokens
while excluding only structural/legal suffix tokens.

## Blocking Decision

Added routes:

- exact compact name and address
- compact name and address prefixes
- country plus significant name token
- country plus significant name-token pair
- country plus address number
- country plus address-number plus address token

Candidate capacity is allocated across exact, name, address, number, and prefix
families before the final per-source cap. This prevents an early common posting
list from consuming the entire candidate budget.

The benchmark must be rerun after these changes before any model training is
started.
