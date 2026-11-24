# Submission ledger — coordinator only

Dashboard: https://arcadegames.hudson-trading.com/challenges/f286a5b8-79df-4c11-94a6-051abfbdcea
Notebook maximum: 10 uploads. Used/remaining: **unverified**. Deadline: **verify dashboard**.
External pretrained models: **unverified**. No uploads performed by this agent; actual dashboard usage remains unverified.

| Attempt | UTC | Source SHA | CSV + SHA256 | Validation/audit Sharpe | Score | Status |
|---|---|---|---|---|---|---|

Before upload: pull main, freeze source, check 10,000 unique expected IDs, exact header,
finite predictions; reserve entry, upload, then record result. Aim T+35 first and T+40 final.

## Early baseline ready — 2026-11-24

User requested an earlier first submission. PR #7 merged at source SHA `8998dc3d748eae68354bc413161bd161987b5c2c`.
Independent reproduction: selection Sharpe **3.122185731987107** on 1,500 sessions; four tests passed.
File: `reports/events/predictions.csv`; 10,000 exact unique OOS IDs, finite values, correct header.
SHA256: `6d070f87bf8c18cbadc4e9fe94cf398df5ea958a8db1393f698a63d7046d065b`. Reproduced values match at rtol=1e-10 / atol=1e-12.

Status: **READY, NOT UPLOADED**. Uses 85% development refit; final audit remains sealed.
First upload can happen immediately after dashboard quota/deadline check; no need to wait
for other lanes or #4. Operator must reserve actual attempt number and report dashboard score.
The later final candidate still follows #4 audit/integration gate.
