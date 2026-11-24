# Live sprint board

[GitHub issues](https://github.com/wryzxec/hrt-algo-arena/issues) are the live status source.
This table records the initial dependency graph; coordinator updates it at merge milestones.
Harness commit: 36aedd3. Local tests and GitHub CI passed on initial scaffold.

| Issue | Owner | Initial state | Dependencies | Exclusive implementation scope |
|---|---|---|---|---|
| [#1 Events](https://github.com/wryzxec/hrt-algo-arena/issues/1) | GlenFilson | ready | harness published | models/events.py, tests/test_events.py, reports/events/ |
| [#2 Prices](https://github.com/wryzxec/hrt-algo-arena/issues/2) | AhmadHNazim | ready | harness published | models/price.py, tests/test_price.py, reports/price/ |
| [#3 Text](https://github.com/wryzxec/hrt-algo-arena/issues/3) | chfmath | ready | harness published | models/text.py, tests/test_text.py, reports/text/ |
| [#4 Integration/audit](https://github.com/wryzxec/hrt-algo-arena/issues/4) | wryzxec | blocked | #1, #2, #3 merged + checks | shared code, ensemble, strategy |
| [#5 Submission](https://github.com/wryzxec/hrt-algo-arena/issues/5) | wryzxec | blocked | #4 completed + #6 dashboard verification | final export, ledger |
| [#6 Coordination](https://github.com/wryzxec/hrt-algo-arena/issues/6) | wryzxec | ready | none | dependencies, docs, assignments, merge queue |

Model paths are under src/arena/. Only assigned owner may work each issue. No claim-by-comment
races: ownership is preassigned, and coordinator alone reassigns after old owner stops.
Blocked tasks remain blocked until coordinator verifies merged prerequisites and passing CI.
At cutoff, coordinator can explicitly skip a missing optional lane and use best completed
baseline; record that decision in #4 before marking ready. No silent dependency bypass.

Conservative provisional UTC schedule on 2026-11-24: experiments stop 19:13, PRs 19:16,
integrate 19:18, first upload 19:23, final 19:28. This is an internal buffer, not a verified
contest deadline. Earlier dashboard deadline wins. Verify it immediately in #6.

FinBERT is optional, not a dependency. No models have measured scores at initial setup.
