# Changelog

Newest first. This recipe serves one DGX Spark. The image is `tensorfold-qwen38:zig-db28187`, built by
`scripts/prepare.sh` from TensorFold `db281878` plus `patches/`.

## Unreleased: engine `37763df` (the two-Spark recipe's v1.1 fixes)

Merged into `zig-single` as `453439d`. `patches/0001`–`0009` are that tree against TensorFold `db281878`.
`zig/src/cuda/graph.zig` ships in `patches/0001-build-registry-server.patch`. The kernel set is unchanged.
Local image `tensorfold-qwen38:zig-db28187` label `tf.patches` is `4144696f21e3`
(`sha256:68bff44f627d2a1fc7dd980354ad7bde22cdf5bd41675f8cc3bc2ba9c0b60edc`; the previous image was `d84d7cc22655`).
The kernel-set check passed: 320 kernels, `KERNEL_SOURCE_MTIME=1791318675`.

### Fixed
- **A server exit under long load.** A CUDA graph capture, instantiate or upload that fails now leaves the round's
  result from its eager run in place (same bits), pauses graph captures for a while (64 rounds, doubling on repeated
  failures, up to 4096), and logs a warning. The server keeps serving. `TF_FLASHNEXT_GRAPH_LOG=1` logs the graph
  counts and a context-wide check before each instantiate. It is off by default. The two-Spark stack hit
  `cuGraphInstantiateWithFlags: CUDA_ERROR_NOT_PERMITTED` after about 15 minutes of heavy load; this server runs the
  same engine.
- **Requests that fill the window exactly were refused**
  ([#1](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/issues/1),
  [#2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2), reported, diagnosed and fixed by
  [321sssrt-bit](https://github.com/321sssrt-bit)): admission counted the draft window against `--context`, so a
  prompt plus `max_tokens` equal to the window got `PromptTooLong`. The engine already keeps those rows beyond the
  window; admission now checks prompt + reply against `--context`.

### Added
- `tools/context_boundary.py` (by [321sssrt-bit](https://github.com/321sssrt-bit), from
  [PR #2](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold/pull/2)): full prompt/reply budgets
  succeed with drafts on and off, one token over the window returns HTTP 400.

## Unreleased: INT4-AutoRound, FP8 KV, image and video

The serving path is TensorFold's Zig engine (`tensorfold-native`, TP=1). The engine is `37763df` (owner build
`20e709a` plus the graph-failure fallback and the full-window admission fix), merged into `zig-single` as `453439d`.
`patches/0001`–`0009` are that tree against TensorFold `db281878`.

### Upgrade from the Python recipe

- **Checkpoint.** Was [Vontra's MLX 4-bit conversion](https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP).
  Now [azampatti's INT4-AutoRound](https://huggingface.co/azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound) at
  `1464274120d36a4d8fcaa934552334a7d83ce0fd` (top-5 experts, 4.8B active).
- **KV.** Was int8. Now FP8 by default (`KV_DTYPE=fp8`): lossy, about 98.8% top-1 agreement with bf16, about 1.8x the
  pool. `KV_DTYPE=bf16` is the exact cache. One 262,144-token sequence is 4.37 GiB in fp8 and 7.52 GiB in bf16. The
  FP8 format is adapted from MiaAI-Lab's GLM recipe patch `0038-glm-kv-fp8`.
- **N-gram table.** The SSD reader is gone. The FP8 table in the checkpoint's `ple-table/` loads with the weights.
- **Draft-language image.** Gone. `DRAFT_LANGUAGE` and `patches/languages/0010` are not applied. Javier
  ([jvr0x](https://github.com/jvr0x)) authored those lists
  ([#84](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/pull/84)).
- **Image and video.** `VISION=1` by default, `VISION_URLS=0`, 50 images, 4 videos, 16,384 image tokens a request
  (at most 4,096 an image), 16,384 video tokens. Adapted from MiaAI-Lab's patches 0008 and 0009 and from TensorFold's
  vision code by Ash Hart and the TensorFold contributors. The helper is TensorFold's Python tree with transformers
  5.17.0, PyAV 19.0.1 and Pillow. FP8 KV and `--vision` run together.
- **Memory floor.** `TENSORFOLD_MEMORY_RESERVE_GIB=10`. `prepare.sh` and `start.sh` refuse a Spark with less than
  10 GiB `MemAvailable`. With vision on, the engine also keeps 2 GiB for the helper
  (`TENSORFOLD_VISION_WORKSPACE_MIB=2048`; the tower peaks around 0.9–1.7 GiB). A request that does not fit the pool
  (prompt plus `max_tokens`) is refused, not queued. Structured outputs (`response_format`, `guided_*`) are refused.
- **Streams.** `PARALLEL` is 8 (a cap, 1 to 16). On spark4 with FP8 KV and vision on, sequence memory was 30.22 GiB,
  which holds six full 262,144-token windows. A request that would not fit is refused. `TF_FLASHNEXT_PRODUCT_STREAMS`
  is 2.

### Earlier Zig boot (still the measured tables in the README)

Engine `6eb39c1`, bf16 KV, text only, spark4: prose 63.3 / 95.4 / 118.5 / 148.6 tok/s at 1 / 2 / 3 / 4 requests.
Sequence memory on that boot was 33.28 GiB. Those tables are replaced when the FP8 + vision boot is measured.
