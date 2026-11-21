# gta6-trophy-forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Regian1117/gta6-trophy-forecaster/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Regian1117/gta6-trophy-forecaster/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Deterministic week-one PlayStation 5 trophy earn-rate forecaster for Grand Theft Auto VI on Exophase (Node 22, standard library only, zero network). Decouples denominator player growth and per-trophy earner accumulation across deduplicated refresh states, applies split Empirical-Bayes group shrinkage (`active` vs `stagnant`), and computes calibrated five-quantile forecasts via exact CDF inversion over a 5-regime Student-t (`df=4`) scenario and `0.01%` hurdle-mixture distribution.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5cmgsxgg&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
