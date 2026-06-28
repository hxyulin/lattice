# lattice-engine

The search and evaluation layer of
[Lattice](https://github.com/hxyulin/lattice) - the part that, given a position,
decides which move to play. Built on [`lattice-board`](../lattice-board), which
owns the rules.

> Status: early scaffold. Search and evaluation land as each is designed.

## Planned

- `eval` - static evaluation (material first, then positional terms).
- `search` - negamax with alpha-beta over `Board::pseudo_legal_moves`, filtering
  legality with make / `is_attacked` / unmake.

Protocol IO is not here - that belongs to [`lattice-uci`](../lattice-uci). This
crate exposes a plain Rust API that the UCI layer (or a test) drives.
