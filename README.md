# Assistant Axis Emergence and Attribution

This project traces an Assistant-Axis-like direction through open pretraining checkpoints and tests whether checkpoint-aligned training sequences exert first-order gradient pressure toward that direction.

The MVP target is:

- Model: `EleutherAI/pythia-410m-deduped`
- Checkpoints: Hugging Face revisions from `step0` through `step143000`
- Primary training stream: `pietrolesci/pile-deduped-pythia-preshuffled`
- Official packed stream fallback: `EleutherAI/pile-deduped-pythia-preshuffled`
- Raw text/source lookup: `EleutherAI/the_pile_deduplicated` and, where useful, `pietrolesci/pile-deduped`

The core claim:

> We trace an Assistant-Axis-like direction through open pretraining checkpoints, identify periods where its geometry stabilizes or changes rapidly, and test whether training sequences in those periods exert first-order gradient pressure toward that direction.

The project does not claim that it has identified the exact documents that created the Assistant Axis. The initial attribution unit is a packed Pythia training sequence, not a clean raw document.

## Project Docs

- [MVP scope](docs/design/mvp_scope.md)
- [Pythia training-data contract](docs/design/pythia_training_data_contract.md)
- [Tasklist](docs/design/tasklist.md)
- [Repo build map](docs/design/repo_build_map.md)
- [Project tracker](docs/design/project_tracker.md)
- [VAST MVP runbook](docs/runbooks/vast_mvp_runbook.md)
- [VAST MVP checklist](docs/runbooks/vast_mvp_checklist.md)
- [Research-engineering curriculum](docs/learning/research_engineering_curriculum.md)

## Artifact Policy

All generated experiment outputs go under `artifacts/runs/...`. Large downloaded model or dataset caches should be configured explicitly and documented in run manifests rather than scattered into ad-hoc paths.
