# Join the sprint

Coordinator: wryzxec. Workers: chfmath, GlenFilson, AhmadHNazim. Assignments are provisional
based on available accounts, not assumed expertise. Only coordinator changes ownership,
and only after old owner acknowledges stopping. One agent per account/issue at a time.

1. Pull main, read AGENTS.md, install/test README setup.
2. `gh issue list --assignee @me --state open`; work only state:ready tasks with merged dependencies.
3. Comment START with branch, ETA and hardware. Replace state:ready with state:working.
4. Branch agent/<login>/issue-<number>-<slug>. Stay within assigned file scope.
5. Post progress at T+10/T+20; report blockers immediately.
6. Push branch; fetch origin and merge origin/main before PR. Never discard teammate changes.
7. PR must link issue and include command, measured Sharpe, runtime and report artifacts.
   Mark state:review. Coordinator pulls newest main and merges passing PRs sequentially.

Comments are not atomic claims. Preassignment is the ownership lock; unassigned agents
request a disjoint issue. Dependencies unblock only after implementation PR merges and
checks pass; a closed issue is insufficient. Only coordinator sets dependent tasks ready.
If optional work is skipped, coordinator records fallback explicitly before unblocking.
These are coordination rules, not enforced branch protection. No shared-file edits by workers:
coordinator owns dependencies, loader/evaluator, strategy, task board and submission ledger.
If pull cannot fast-forward, preserve work and report; no resets or force pushes.

Coordinator serializes uploads in docs/SUBMISSIONS.md with reserved attempt, source SHA,
CSV hash and result. Check quota first. Interrupted upload is consumed until dashboard
confirms otherwise. Workers never upload. Do not spend all ten attempts searching weights.

T0 is user's 45-minute instruction, not each agent's start. Tracker records UTC cutoff.
T+25 stop experiments; T+30 merge/freeze; T+35 first upload; T+40 final upload/check;
T+45 hard stop. Use best completed lane if others are late.
