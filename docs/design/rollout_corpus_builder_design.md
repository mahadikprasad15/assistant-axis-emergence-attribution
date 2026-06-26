# Rollout Corpus Builder Design

`RolloutCorpusBuilder` is the first real script in the repo. Its job is deliberately narrow: turn the readable rollout config into a fixed JSONL corpus plus a manifest.

## Input

```text
configs/rollouts/assistant_axis_roles_v0.yaml
configs/rollouts/assistant_axis_source_material_v0.yaml
```

## Output

```text
data/rollouts/assistant_axis_rollouts_v0.jsonl
data/rollouts/assistant_axis_rollouts_v0_manifest.json
```

## Script

```text
scripts/rollouts/build_rollout_corpus.py
```

## What It Does

1. Load the rollout config and source-material config.
2. Validate the v0 target shape:
   - 48 roles,
   - 16 roles per role group,
   - 20 selected questions,
   - 4 default prompt families.
3. Build role records:
   - one record for each role/question pair,
   - currently first instruction variant only,
   - expected count: 960.
4. Build default records:
   - one record for each default prompt/question pair,
   - expected count: 80.
5. Write the JSONL corpus.
6. Write a manifest with config hashes, counts, warnings, and output hash.

## What It Does Not Do

- It does not call a model.
- It does not generate final response text.
- It does not extract activations.
- It does not build the Assistant Axis.
- It does not use old trait-instruction conditions.

## Important Current Warning

The rollout config has 48 target roles, but only a subset of role instructions are locally imported from the older trait-geometry repo. Roles marked `planned_upstream_import` are readable placeholders until we either source-verify them from Assistant Axis or explicitly decide to keep locally drafted role prompts.

## Learning Walkthrough

For a learning-first explanation of the configs, schemas, helper functions, and generated artifacts, read:

```text
docs/learning/rollout_corpus_walkthrough.md
```
