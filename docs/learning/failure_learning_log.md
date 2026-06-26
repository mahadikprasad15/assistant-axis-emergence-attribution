# Failure Learning Log

This log records failures, near-misses, and reusable prevention rules for this repo. Add entries whenever we encounter a bug, confusing behavior, bad assumption, or known failure from a related repo that could recur here.

The point is not blame. The point is to make future scripts safer.

## Entry Template

```text
Category:
Component:
Symptom:
Root cause:
Prevention:
Checks to add:
Source/evidence:
Status:
```

## Activation and Generation Span Failures

### FL-001: Generation Padding Can Corrupt Completion Decoding

Category: tokenization / padding / generation

Component: future fixed-response generation runner

Symptom:

- Batched generation can produce confusing or empty decoded completions if the script slices output tokens using the wrong input width.
- Decoder-only generation appends new tokens after the padded input width, not after each unpadded prompt length.

Root cause:

- Generation batching and activation caching need different padding policies.
- In the earlier trait-geometry repo, the durable lesson was: use left padding for generation so all prompts in the batch share one input width and completion decoding can use the padded input width.

Prevention:

- For generation, set `tokenizer.pad_token = tokenizer.eos_token` if no pad token exists.
- For generation, set `tokenizer.padding_side = "left"`.
- Decode completions from the padded batch input width, not from each unpadded prompt length.
- Record generation batch size and tokenizer padding side in the manifest.
- If a batch fails, record every rollout id in the failed batch.

Checks to add:

- Assert every generated response has non-whitespace text unless the run explicitly allows empty responses.
- Count empty responses in the generation manifest.
- Save `prompt_token_count`, `generated_token_count`, and `decoded_response_char_count`.
- Fail main activation runs if any required fixed response is empty.

Source/evidence:

- Earlier repo note: `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/docs/learning/research_engineering_curriculum.md`, lines 618-637.

Status:

- Imported as a prevention rule before implementing fixed response generation in this repo.

### FL-002: Activation Caching Needs Right Padding and Per-Item Response Spans

Category: tokenization / padding / activation caching

Component: future `ActivationCacheRunner`

Symptom:

- Response-token pooling can silently pool the wrong positions if batched padding shifts real text positions.
- Mean activations can be wrong even if tensor shapes look valid.

Root cause:

- Activation caching is not generation. We already have full text and need to read residual-stream activations over the real response span.
- In the earlier trait-geometry repo, activation caching used right padding so padding appears after real text and does not shift each row's real response span.

Prevention:

- For activation caching, use right padding.
- Compute `prompt_len` and `full_len` per item before batching.
- Run the model/cache over the right-padded full-text batch.
- For each row, pool only:

```text
cache[name][row_idx, prompt_len:full_len, :]
```

- Preserve one activation artifact and one index row per rollout id, even if the forward pass is batched.

Checks to add:

- Assert `full_len > prompt_len` for every record.
- Assert `response_token_count = full_len - prompt_len > 0`.
- Assert pooled tensor shape equals `[hidden_size]`, currently `[1024]` for Pythia-410M.
- Record `padding_side`, `prompt_token_count`, `full_token_count`, `response_token_count`, `response_token_start`, and `response_token_end` in the activation index.
- Add a small smoke fixture with two different prompt lengths in one batch to test span correctness.

Source/evidence:

- Earlier repo note: `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/docs/learning/research_engineering_curriculum.md`, lines 639-660.
- Earlier repo implementation: `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/src/trait_geometry/activations/cache_activations.py`, lines 335-386.

Status:

- Imported as a prevention rule before implementing activation caching in this repo.

### FL-003: Empty Responses Must Block Response-Token Mean Pooling

Category: empty outputs / response spans / activation caching

Component: fixed-response generation and future `ActivationCacheRunner`

Symptom:

- Activation caching over response tokens can fail or produce empty means when a generated response is empty or tokenizes to no additional tokens.

Root cause:

- `response_token_mean` requires at least one response token.
- The earlier activation cache code explicitly raised an error when `full_len <= prompt_len`.

Prevention:

- Treat empty response text as invalid for main runs.
- Treat `full_len <= prompt_len` as a hard error for `response_token_mean`.
- Keep failed rollout ids in status/progress so they can be inspected and regenerated.

Checks to add:

- In response generation/import validation: reject empty responses.
- In activation caching: raise if `full_len <= prompt_len`.
- In the activation manifest: report min/median/max response token counts.
- In the activation inspector: list the shortest response spans.

Source/evidence:

- Earlier repo implementation: `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/src/trait_geometry/activations/cache_activations.py`, lines 285-291 and 357-366.

Status:

- Imported as a prevention rule before implementing fixed-response generation and activation caching in this repo.
- Implemented in `scripts/rollouts/import_fixed_responses.py` for imported response text; still required in the future generator runner and activation cache runner.

### FL-004: Resume Logic Must Check Both Index Rows and Artifact Files

Category: resume / artifact integrity / activation caching

Component: future `ActivationCacheRunner`

Symptom:

- A resumed activation run can skip work incorrectly if it trusts progress metadata without checking that the activation artifact actually exists.

Root cause:

- Progress files and index rows can exist even when a previous run failed before all tensor artifacts were written.
- The earlier activation runbook says completed activation artifacts are skipped only when the activation artifact path exists.

Prevention:

- On resume, read the activation index and only skip rollout ids whose `activation_path` exists.
- Keep `progress.json` as a cursor/checkpoint, but treat artifact existence as the final completion check for each unit.
- Allow rerun with the same run root and smaller batch size after OOM.

Checks to add:

- Startup audit: count index rows, existing activation files, and missing activation files.
- If an index row points to a missing file, mark that rollout id incomplete.
- Write resume summary to `logs/run.log`.

Source/evidence:

- Earlier repo runbook: `/Users/prasadmahadik/Documents/Do traits remain consistent across personas?/docs/design/vast_activation_runbook.md`, lines 91-103.

Status:

- Imported as a prevention rule before implementing activation caching in this repo.

### FL-005: Broken Local PyTorch Install Blocks Llama and Pythia Runs

Category: dependency-config / model runtime

Component: `FixedResponseGenerator` with `hf_local`, future `ActivationCacheRunner`

Symptom:

- `transformers` and `torch` appear importable from module discovery, but importing `torch` fails at runtime.
- The Llama local-files smoke run fails before model/tokenizer loading can begin.

Root cause:

- The active `python3` environment has an incomplete or broken PyTorch binary install.
- The missing dynamic library is:

```text
libtorch_cpu.dylib
```

Prevention:

- Add a model-runtime preflight before any expensive generation or activation run.
- The preflight must actually `import torch`, not only check `importlib.util.find_spec("torch")`.
- Keep failed run status durable so a model-load failure does not leave `meta/status.json` as `running`.

Checks to add:

- Add a `scripts/system/check_model_runtime.py` or equivalent preflight.
- Check `torch.__version__`, `transformers.__version__`, `torch.cuda.is_available()` or MPS availability, and a tiny tensor operation.
- Run the preflight before Llama generation and Pythia activation caching.

Source/evidence:

- Failed command:

```bash
python3 scripts/rollouts/generate_fixed_responses.py \
  --provider hf_local \
  --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
  --variant llama-3.2-1b-instruct \
  --run-id llama-localfiles-smoke-v0 \
  --limit 1 \
  --local-files-only \
  --max-new-tokens 32
```

- Error excerpt:

```text
ImportError: Library not loaded: @rpath/libtorch_cpu.dylib
```

Status:

- Encountered during the first Llama local-files smoke test.
- Generator and activation runners were patched to mark failed runs explicitly when model/runtime loading raises.
- Resolved by creating repo-local `.venv` and installing the model stack recorded in `requirements-model.txt`.

### FL-006: Gated Hugging Face Model Requires Auth Before Llama Generation

Category: security-access / dependency-config

Component: `FixedResponseGenerator` with `hf_local`

Symptom:

- Runtime preflight passes in the repo-local `.venv`.
- Llama generation still fails while loading `meta-llama/Llama-3.2-1B-Instruct`.
- Hugging Face returns a 401 gated-repo error.

Root cause:

- `meta-llama/Llama-3.2-1B-Instruct` is a gated Hugging Face repository.
- The local Hugging Face CLI reports:

```text
Error: Not logged in
```

Prevention:

- Check Hugging Face auth before attempting gated model downloads.
- Use `hf auth whoami` in the same environment used for generation.
- Keep gated model access as a separate blocker from Python/PyTorch runtime health.

Checks to add:

- Add a generation preflight that records `hf auth whoami` status when the model id starts with `meta-llama/`.
- Fail fast with a clear message if not logged in or if access to the gated model has not been granted.

Source/evidence:

- Failed command:

```bash
.venv/bin/python scripts/rollouts/generate_fixed_responses.py \
  --provider hf_local \
  --hf-model-id meta-llama/Llama-3.2-1B-Instruct \
  --variant llama-3.2-1b-instruct \
  --run-id llama-3.2-1b-smoke-v1 \
  --limit 1 \
  --max-new-tokens 32
```

- Error excerpt:

```text
You are trying to access a gated repo.
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted.
You must have access to it and be authenticated to access it.
```

Status:

- Current blocker after runtime repair.
- Requires `hf auth login` with an account that has accepted the Llama model terms, or a switch to an ungated instruction model.

## Open Follow-Ups

- Add these checks to the future fixed-response generator.
- Add these checks to the future activation cache runner.
- Add a tiny span-alignment unit test or smoke script before running the full 1040-record corpus.
- Add a model-runtime preflight before running Llama generation or Pythia activation caching.
- Add a Hugging Face gated-model auth preflight before Llama generation.
