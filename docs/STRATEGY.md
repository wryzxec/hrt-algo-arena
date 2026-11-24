# Best plan within 45 minutes

Prioritize cheap supervised text features and observed price response. This is a hypothesis,
not a demonstrated optimum. Synthetic headline templates may favor TF-IDF over general
financial sentiment. Good news can already be priced in by bar 49.

| Lane | Owner | Deliverable | Target |
|---|---|---|---|
| Text | chfmath | TF-IDF 1–2 grams + Ridge; all vs recent headlines | first T+12, final T+25 |
| Events | GlenFilson | event phrases, numbers, recency, observed response | first T+15, final T+25 |
| Prices | AhmadHNazim | momentum/reversal/volatility + Ridge or small trees | first T+12, final T+25 |
| Integration | wryzxec | compare/blend/audit/export | freeze T+30, upload T+35 |

Text: small alpha grid (1,10,100), train-only vocabulary. Try recent headlines separately.
Events: inspect TRAIN headlines only; extract positive/negative event phrases, magnitudes,
negation, counts, latest event and decay half-lives 5/15 bars. Include price move since the
headline using only available bars. Avoid constructing encodings from validation targets.
Prices: return windows 1/3/5/10/20/49, volatility, high-low range, distance from extrema,
last candle. Compare Ridge and small HistGradientBoostingRegressor. No extensive tuning.

FinBERT is optional and deprioritized. It predicts sentiment, not forward returns. Only
attempt after a working baseline, with external-model rules verified and spare runtime.
Never fine-tune in this sprint. Benchmark 256 texts, estimate full train/OOS runtime;
abort if not complete by T+25. Use frozen probabilities, exact-text caching and batching,
recency aggregates, and supervised calibration to returns. Keep only if held-out results help.

Use common 70/15/15 split. Compare identical selection sessions; log every trial. Coordinator
tries best single model and at most one equal-weight blend, normalizing components by
training prediction scale. Inspect PnL tails and concentration. Freeze then open audit once;
do not tune repeatedly against it. Refit chosen model on all labels only after audit decision.
Retain a working fallback; missing lanes never block a valid upload.

Sharpe is invariant to positive global scale. The unconstrained population-optimal position
is proportional to E[r|x]/E[r²|x], but estimating this can overfit. Prefer raw predictions
first; at most test conservative clipping/volatility scaling on development data. No future
bars/headlines, no validation-target centering, no leaderboard weight searches.

Inspiration:
- [Two Sigma: Using News to Predict Stock Movements (2018)](https://www.kaggle.com/competitions/two-sigma-financial-news):
  close news/market prediction precedent. Borrow multimodal features and validation discipline;
  its data, API and metric are not interchangeable with ours.
- [XForecast KDD 2026 paper](https://arxiv.org/abs/2610.16131): recent Kaggle-hosted news+prices
  challenge, four-week target and weighted hit rate. Live status not verified.
- [ProsusAI FinBERT](https://huggingface.co/ProsusAI/finbert): financial sentiment classifier
  with positive/negative/neutral outputs. No evidence yet it improves this synthetic task.

Notebook supplies current spec. Authenticated challenge page was inaccessible to browser tool.
Verify dashboard deadline, remaining uploads, external model rules and official std convention.
No measured model performance yet at setup.
