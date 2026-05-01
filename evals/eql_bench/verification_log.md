# EQL-Bench v2 Verification Log

**Date:** 2026-04-30
**Sample:** 370 scenarios (100% of 370 total)
**Pass rate:** 361/370 (97.6%)

## Criterion Results

| Code | Description | Pass | Fail |
|---|---|---|---|
| C1 | uncertain_statement present | 370 | 0 |
| C2 | value_fragment matches statement | 370 | 0 |
| C3 | qualifier in statement (or ghost) | 369 | 1 |
| C4 | valid domain | 370 | 0 |
| C5 | valid qualifier_type | 370 | 0 |
| C6 | qualifier_fragment ≤ 40 chars | 370 | 0 |
| C7 | value_fragment specific | 370 | 0 |
| C8 | statement ≥ 30 chars | 362 | 8 |

## Failing Scenarios

| ID | Domain | Failed Criteria | Issue |
|---|---|---|---|
| cpl-v2-029 | compliance | C3 | No qualifier fragment or synonym found in statement (frags: ['vendor says', "haven't reviewed spec", 'unverified', 'vendor claim', 'implementation details unknown']) |
| ghost-api-002 | api | C8 | statement too short (29 chars) |
| ghost-api-003 | api | C8 | statement too short (29 chars) |
| ghost-api-004 | api | C8 | statement too short (29 chars) |
| ghost-dbg-004 | debug | C8 | statement too short (27 chars) |
| ghost-dbg-006 | debug | C8 | statement too short (28 chars) |
| ghost-des-001 | design | C8 | statement too short (27 chars) |
| ghost-des-002 | design | C8 | statement too short (22 chars) |
| ghost-des-005 | design | C8 | statement too short (26 chars) |

## Acceptance Decision

Pass rate: **97.6%** (threshold: 95%)

✓ ACCEPTED — EQL-Bench v2 is publication-ready.