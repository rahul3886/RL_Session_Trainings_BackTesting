# Lens 1 Retrain Comparison

This report compares the retrained Lens 1 checkpoints against the
pre-retrain baseline checkpoints on the validation and test splits.

## Test Split

| Model | Rows | Combined Sweep Recall | High Sweep Recall | Low Sweep Recall | Total Sweeps | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retrained_best | 132 | 77.4% | 86.7% | 68.8% | 31 | 10 |
| retrained_final | 132 | 67.7% | 46.7% | 87.5% | 31 | 10 |
| baseline_best | 132 | 58.1% | 80.0% | 37.5% | 31 | 10 |
| baseline_final | 132 | 54.8% | 73.3% | 37.5% | 31 | 10 |

Best on test: `retrained_best` at 77.4%.

- `retrained_best` dominant scenarios: {'S3S4_Continuation': 8, 'S1_BullContinuation': 1, 'S7_HighLiqGrab': 1}
- `retrained_final` dominant scenarios: {'S3S4_Continuation': 5, 'S1_BullContinuation': 2, 'S8_LowLiqGrab': 1, 'S7_HighLiqGrab': 1, 'Mixed/Ranging': 1}
- `baseline_best` dominant scenarios: {'S3S4_Continuation': 7, 'Mixed/Ranging': 1, 'S1_BullContinuation': 1, 'S2_BearContinuation': 1}
- `baseline_final` dominant scenarios: {'S3S4_Continuation': 7, 'S8_LowLiqGrab': 2, 'S1_BullContinuation': 1}

## Validate Split

| Model | Rows | Combined Sweep Recall | High Sweep Recall | Low Sweep Recall | Total Sweeps | Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retrained_final | 130 | 73.7% | 33.3% | 92.3% | 19 | 10 |
| baseline_best | 130 | 63.2% | 66.7% | 61.5% | 19 | 10 |
| baseline_final | 130 | 31.6% | 83.3% | 7.7% | 19 | 10 |
| retrained_best | 130 | 31.6% | 16.7% | 38.5% | 19 | 10 |

Best on validate: `retrained_final` at 73.7%.

- `retrained_final` dominant scenarios: {'S3S4_Continuation': 8, 'S2_BearContinuation': 1, 'Mixed/Ranging': 1}
- `baseline_best` dominant scenarios: {'S3S4_Continuation': 10}
- `baseline_final` dominant scenarios: {'S3S4_Continuation': 9, 'S8_LowLiqGrab': 1}
- `retrained_best` dominant scenarios: {'S3S4_Continuation': 8, 'S2_BearContinuation': 1, 'Mixed/Ranging': 1}
