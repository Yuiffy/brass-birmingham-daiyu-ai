# World Model Evaluation

Dataset: 10000 complete games; held out: 100 games / 12800 transitions.

Current player accuracy: 96.51%
Change precision / recall / F1: 93.43% / 91.31% / 92.36%

| Field group | MAE in native units | Rounded accuracy | MAE on changed fields |
|---|---:|---:|---:|
| cards | 0.008 | 99.30% | 0.2240363508462906 |
| coalMarket | 0.007 | 99.66% | 0.04304874315857887 |
| currentPlayer | 0.017 | 98.25% | 0.0325113870203495 |
| era | 0.000 | 100.00% | 0.0 |
| flip | 0.002 | 99.77% | 0.4240342378616333 |
| income | 0.153 | 96.12% | 1.770206093788147 |
| industry | 0.004 | 99.84% | 0.2342384308576584 |
| ironMarket | 0.002 | 99.97% | 0.024246882647275925 |
| merchant | 0.001 | 99.90% | 0.43923667073249817 |
| money | 0.861 | 75.48% | 2.8039543628692627 |
| network | 0.002 | 99.91% | 0.22155968844890594 |
| resources | 0.004 | 99.71% | 0.2008102536201477 |
| supply | 0.001 | 99.98% | 0.02640032209455967 |
| turn | 0.227 | 93.06% | 1.0620230436325073 |
| vp | 0.093 | 98.34% | 5.282472610473633 |

| Rollout depth | Normalized MAE | Persistence MAE |
|---|---:|---:|
| 1 | 0.00046 | 0.00455 |
| 2 | 0.00127 | 0.00936 |
| 3 | 0.00176 | 0.01357 |
| 4 | 0.00453 | 0.01742 |
| 5 | 0.00492 | 0.02111 |

See evaluation.json for action breakdowns, sample field differences and limitations.