# NNUE pipeline

End-to-end path from datagen text to an embedded network. Validated on macOS
(Metal); the only change for the training box is the backend feature.

## 1. Convert data (text -> bulletformat)

Datagen emits `<FEN> | <score> | <result>` (white-relative cp, result
1.0/0.5/0.0). bullet's loader wants the packed `ChessBoard` binary:

```
cd tools/convert
cargo run --release -- <data.txt> <out.data>        # add --limit N for a slice
```

## 2. Train (bullet)

Architecture is `(768 -> 256) x2 -> 1`, SCReLU, QA=255 QB=64, scale 400 - these
must match `src/board/nnue.rs`.

```
cd tools/trainer
# macOS:
cargo run --release --features metal -- <out.data> <superbatches> <batches_per_superbatch>
# NVIDIA (set CUDA_PATH):
cargo run --release --features cuda  -- <out.data> <superbatches> <batches_per_superbatch>
```

For a real net use the full data and a proper schedule, e.g. `... 40 6104`
(edit the LR/WDL schedule in `src/main.rs` as needed). The quantised network is
written to `checkpoints/lattice-<sb>/quantised.bin`.

If cargo can't clone the bullet git dependency (`SecureTransport error: bad
MAC`), prefix with `CARGO_NET_GIT_FETCH_WITH_CLI=true`.

## 3. Deploy the net

```
cp tools/trainer/checkpoints/lattice-<sb>/quantised.bin src/board/net.nnue
cargo build --release --features nnue
```

The engine embeds `src/board/net.nnue` via `include_bytes!`. `quantised.bin` is
padded to a 64-byte multiple (e.g. 394816 for HIDDEN=256); the loader reads the
394754-byte prefix and ignores the padding.

The committed `src/board/net.nnue` is a weak bootstrap net (small smoke-test
run) so the `nnue` feature compiles - replace it with the real trained net.
