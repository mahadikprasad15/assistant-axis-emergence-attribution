# Gradient Attribution Diagnostics

Before scaling attribution, the smoke pipeline adds four checks:

1. Named fixed targets distinguish native, endpoint, and final axes.
2. Token-level cosine summaries expose cancellation hidden by cosine-after-mean pooling.
3. Optional token-score tensors preserve the full diagnostic signal.
4. A run comparator checks `auto` versus `float32` numerical agreement on identical samples.

For a `step256 -> step512` window, the recommended targets are:

```text
native_step256   = step256 AA
endpoint_step512 = step512 AA
final_step143000 = final AA
```

The fixed endpoint and final targets support paired checkpoint comparisons. The native target answers a different, checkpoint-relative question and must not be substituted into a paired delta without noting that the target changed.
