# Lattice NNUE trainer

This standalone crate pins Bullet and consumes Nyquist's Bullet 1.8 shards.
It is intentionally outside the engine package so GPU dependencies never enter
normal Lattice builds.

On Apple Silicon, the default backend is Metal:

```sh
cargo run --release -- DATA OUTPUT smoke VALIDATION_DATA
```

On an NVIDIA training host:

```sh
cargo run --release --no-default-features --features cuda -- DATA OUTPUT full VALIDATION_DATA
```

Pass multiple Nyquist shards as one comma-separated first argument. Paths must
not themselves contain commas.

Wrap a saved Bullet checkpoint for the engine:

```sh
cargo run --release --bin pack-network -- CHECKPOINT/quantised.bin ../../networks/lattice.nnue
```

The default architecture is `halfkp`. Pass `chess768` as the final argument
when packing the compact absolute piece-square network:

```sh
cargo run --release --bin pack-network -- \
  CHECKPOINT/quantised.bin ../../networks/lattice.nnue chess768
```

Before retraining or SPRT, audit the saved floating graph, quantised checkpoint,
packed engine file, feature ABI, and embedded engine inference together:

```sh
LATTICE_NNUE_PATH=/absolute/path/to/network.nnue cargo run --release \
  --no-default-features --features audit --bin audit-network -- \
  CHECKPOINT/raw.bin CHECKPOINT/quantised.bin /absolute/path/to/network.nnue \
  /absolute/path/to/validation.data 20000
```

`LATTICE_NNUE_PATH` is compile-time input to the engine dependency and must name
the same packed network passed as the third argument.

Run the controlled legacy-recipe comparison (six one-epoch checkpoints, WDL
weight 0.40, stepped `0.001 -> 0.0003` learning rate, fixed seed):

```sh
cargo run --release -- TRAIN_DATA[,TRAIN_DATA...] CHESS_OUTPUT controlled-chess768
cargo run --release -- TRAIN_DATA[,TRAIN_DATA...] HALFKP_OUTPUT controlled-halfkp
```

Score an unwrapped Bullet checkpoint against an independent shard:

```sh
cargo run --release --bin eval-checkpoint -- \
  chess768 CHECKPOINT/raw.bin CHECKPOINT/quantised.bin VALIDATION_DATA
```

Use `halfkp` instead of `chess768` for the other architecture. The evaluator
reports Bullet floating loss, packed-integer loss, and quantisation error under
the controlled 40% WDL objective.
