# World Model Evaluation

Dataset: 400 complete games; held out: 40 games / 4320 transitions.

Current player accuracy: 84.14%
Change precision / recall / F1: 61.66% / 75.94% / 68.06%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.041 | 97.01% | 0.6297327280044556 |
| coalMarket | 0.145 | 84.54% | 0.06107044965028763 |
| currentPlayer | 0.079 | 92.07% | 0.13328564167022705 |
| era | 0.007 | 99.26% | 0.4749999940395355 |
| flip | 0.008 | 99.22% | 0.6053892374038696 |
| income | 0.188 | 95.00% | 3.0434343814849854 |
| industry | 0.009 | 99.56% | 0.661217212677002 |
| ironMarket | 0.060 | 94.54% | 0.5470259189605713 |
| merchant | 0.005 | 99.50% | 0.7316992282867432 |
| money | 3.614 | 64.50% | 9.661490440368652 |
| network | 0.012 | 99.46% | 0.6154865026473999 |
| resources | 0.017 | 98.55% | 0.37089601159095764 |
| supply | 0.005 | 99.66% | 0.09640341252088547 |
| turn | 0.781 | 75.83% | 2.820343017578125 |
| vp | 0.651 | 97.93% | 35.37324905395508 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.00574 | 0.00788 |
| 2 | 0.00728 | 0.01076 |
| 3 | 0.01137 | 0.01501 |
| 4 | 0.01291 | 0.01767 |
| 5 | 0.01803 | 0.02152 |

See evaluation.json for action breakdowns, sample field differences and limitations.