# World Model Evaluation

Dataset: 1000 complete games; held out: 100 games / 12800 transitions.

Current player accuracy: 21.34%
Change precision / recall / F1: 32.66% / 80.07% / 46.39%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.084 | 95.67% | 0.5099577903747559 |
| coalMarket | 0.218 | 89.87% | 0.3157329857349396 |
| currentPlayer | 0.455 | 64.34% | 0.3330123722553253 |
| era | 0.002 | 100.00% | 0.08536417037248611 |
| flip | 0.027 | 99.58% | 0.7046452760696411 |
| income | 0.810 | 43.83% | 2.02143931388855 |
| industry | 0.098 | 97.49% | 0.7872037291526794 |
| ironMarket | 0.190 | 93.80% | 0.2851763069629669 |
| merchant | 0.011 | 99.91% | 0.6843776702880859 |
| money | 6.057 | 5.99% | 7.379394054412842 |
| network | 0.068 | 98.99% | 0.6132595539093018 |
| resources | 0.065 | 98.38% | 0.4458644688129425 |
| supply | 0.124 | 98.28% | 0.4511065185070038 |
| turn | 1.295 | 65.04% | 2.04481840133667 |
| vp | 0.575 | 80.88% | 8.845932006835938 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.01980 | 0.00459 |
| 2 | 0.02986 | 0.00962 |
| 3 | 0.03964 | 0.01368 |
| 4 | 0.05184 | 0.01767 |
| 5 | 0.06065 | 0.02137 |

See evaluation.json for action breakdowns, sample field differences and limitations.