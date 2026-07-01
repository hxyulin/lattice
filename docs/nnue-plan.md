# NNUE Plan (net #1)

Path from the hand-crafted eval (HCE) to a first trained NNUE, kept behind a
cargo feature so HCE stays the default build until NNUE is proven stronger.
Deliberately the smallest architecture that works: the goal of net #1 is to
de-risk the whole pipeline (data -> trainer -> quantized net -> Rust inference
-> SPRT), not to be maximal.

## Locked decisions

- Feature set: **768** = 64 squares x 6 piece types x 2 colors (piece-on-square,
  no king buckets).
- Topology: **(768 -> 256) x2 -> 1**, perspective net (side-to-move and opponent
  accumulators concatenated), **SCReLU** activation, single output bucket.
- Training target: `(1 - lambda) * sigmoid(score / K) + lambda * wdl`, lambda ~= 0.4.
- Quantization: feature transformer i16, output i8 (`QA = 255`, `QB = 64`).
- Toggle: cargo `--features nnue`. The HCE build stays byte-identical to today's
  engine; the NNUE build carries no HCE. Delete HCE once NNUE wins (as Stockfish
  did in SF16).
- Trainer: **bullet** on the RTX 3070 Ti laptop (CUDA). Not Mac MPS, not the AMD
  GPU.
- Data: start with the current ~30M positions. Volume, not depth, drives net #1.

## Machines

- M3 Pro (this Mac): datagen + engine dev. Not training.
- i9-12900K + RTX 3070 Ti: **training** (bullet, CUDA).
- Ryzen 9950X3D + RX 9070 XT: datagen / spare CPU. Skip RDNA4 GPU training (ROCm
  immature).
- Nvidia desktop (available in 1-2 days): switch training here later if faster.

## Pipeline

Datagen already emits text rows `<fen> | <score> | <wdl>` (White-relative). The
trainer wants `bulletformat` (packed binary). A small Rust converter bridges them.

1. **Converter** (`text -> bulletformat`). Use the `bulletformat` crate's
   `ChessBoard` packer. Round-trip test: a packed then unpacked position must
   reproduce the same board + score + wdl (mirrors the Texel feature test).
2. **Train** `768->256->1` in bullet on 30M. A few hundred superbatches; nets are
   tiny, so this is minutes on the 3070 Ti. Output `net.nnue` (quantized).
3. **Rust inference** in `src/engine/` behind `--features nnue`:
   - Incremental accumulator maintained in `board.rs` on make/unmake. This reuses
     the *exact* hook pattern the tapered PeSTO accumulator already uses
     (`put_piece`/`remove_piece`) - just a wider vector instead of two scalars.
   - Forward pass: accumulator -> SCReLU -> output layer -> scale to centipawns.
   - Load the quantized net (`include_bytes!` embed).
4. **Anti-divergence gate (hard milestone).** The Rust eval must match bullet's
   eval on a batch of sampled FENs to within the quantization tolerance *before*
   any SPRT. This catches feature-indexing and quantization bugs that a game
   result would only show as unexplained weakness.
5. **Bench + SPRT** NNUE build vs HCE main. Expect a large positive (a first net
   is typically +100-300 Elo over HCE).
6. **Reinforce.** Regenerate data with the stronger net, retrain -> net #2.
   Repeat. This is where the 30M -> 100M+ gap closes.

## Integration notes

- The accumulator is the crux and is half-built: Lattice already updates an
  incremental White-relative eval in `board.rs`. The NNUE accumulator slots into
  the same make/unmake seams. Get correctness first (recompute-from-scratch equals
  incremental, a test the board already has for PeSTO), then optimize with SIMD.
- Keep `evaluate()` as the HCE path under the default build; the `nnue` feature
  swaps in the network eval. One clear seam, no runtime branching.

## Open choices (defaulted, adjust if desired)

- HL size 256 (vs 128 faster / 512 stronger).
- Feature set 768 (vs king-bucketed HalfKA - stronger, more integration risk).
- SIMD: skip for net #1 (correctness first), add once it passes.
