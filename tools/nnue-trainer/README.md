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
