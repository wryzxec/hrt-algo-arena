# Live sprint board

[GitHub issues](https://github.com/wryzxec/hrt-algo-arena/issues) are the live status source.
This table records the initial dependency graph; coordinator updates it at merge milestones.
Harness commit: 36aedd3. Local tests and GitHub CI passed on initial scaffold.

| Issue | Owner | Latest recorded state | Dependencies | Exclusive implementation scope |
|---|---|---|---|---|
| [#1 Events](https://github.com/wryzxec/hrt-algo-arena/issues/1) | GlenFilson | merged (#7) | harness published | models/events.py, tests/test_events.py, reports/events/ |
| [#2 Prices](https://github.com/wryzxec/hrt-algo-arena/issues/2) | AhmadHNazim | ready | harness published | models/price.py, tests/test_price.py, reports/price/ |
| [#3 Text](https://github.com/wryzxec/hrt-algo-arena/issues/3) | chfmath | review (#8); fit filtering and audit provenance requested | harness published | models/text.py, tests/test_text.py, reports/text/ |
| [#4 Integration/audit](https://github.com/wryzxec/hrt-algo-arena/issues/4) | wryzxec | blocked | #1, #2, #3 merged + checks | shared code, ensemble, strategy |
| [#5 Submission](https://github.com/wryzxec/hrt-algo-arena/issues/5) | wryzxec | blocked | #7 baseline verified + dashboard check; #4 for final | final export, ledger |
| [#6 Coordination](https://github.com/wryzxec/hrt-algo-arena/issues/6) | wryzxec | working; dashboard verification outstanding | none | dependencies, docs, assignments, merge queue |

Model paths are under src/arena/. Only assigned owner may work each issue. No claim-by-comment
races: ownership is preassigned, and coordinator alone reassigns after old owner stops.
Blocked tasks remain blocked until coordinator verifies merged prerequisites and passing CI.
At cutoff, coordinator can explicitly skip a missing optional lane and use best completed
baseline; record that decision in #4 before marking ready. No silent dependency bypass.

Conservative provisional UTC schedule on 2026-11-24: experiments stop 19:13, PRs 19:16,
integrate 19:18, first upload 19:23, final 19:28. This is an internal buffer, not a verified
contest deadline. Earlier dashboard deadline wins. Verify it immediately in #6.

FinBERT is optional, not a dependency. At 19:03 UTC, text PR reports selection Sharpe 3.477
with volatility sizing; merge waits for review/provenance. Price owner was asked to
acknowledge START/ETA; ownership stays unchanged until an acknowledged handoff.

Early-baseline update: #7 merged and independently reproduced (selection Sharpe 3.12219).
`reports/events/predictions.csv` is ready now; first upload no longer waits for 19:23 or
other lanes. Actual upload remains pending dashboard access/quota verification. Final audit
remains untouched. See SUBMISSIONS.md and #5 for explicit early-path dependency change.
