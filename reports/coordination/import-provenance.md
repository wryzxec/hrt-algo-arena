# Imported model provenance follow-up

User explicitly extended the experiment cutoff on 2026-11-24. This issue #6 change fixes metadata only; model configurations are unchanged and audit labels were not scored.

An independent PR #11 reproduction imported its model through PYTHONPATH from an archived checkout while running in the main working directory. The prior evaluator recorded only working-directory source hashes, omitting the actual candidate. New `imported_source` metadata records resolved paths and SHA256 hashes for the actual model and loaded arena modules, including the executing evaluator. `git_context_directory` identifies what the existing Git metadata describes. Historical directory hashes remain available. External model helper modules outside the arena package and in-memory mutations are not exhaustively captured.

Validation: 22 tests passed. Regression coverage imports a candidate outside the working directory and verifies the actual imported file hash. Events reproduction: selection Sharpe 3.122185731987107, runtime 2.99 seconds, 10,000 predictions; CSV SHA256 6d070f87bf8c18cbadc4e9fe94cf398df5ea958a8db1393f698a63d7046d065b, unchanged from the merged baseline. This run used uncommitted source, explicitly marked in its metrics. A subsequent metadata-only fix includes the evaluator when invoked as __main__; all 22 tests passed again.

Reproduce with `.venv/bin/python -m arena.evaluate --model arena.models.events --out artifacts/import-provenance-events-02` and `.venv/bin/python -m pytest -q`. Local run artifacts: artifacts/import-provenance-events-01/.
