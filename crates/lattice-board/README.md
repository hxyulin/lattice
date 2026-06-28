# lattice-board

The core of the [Lattice](https://github.com/hxyulin/lattice) chess engine:
board representation, pieces, squares, moves, and pseudo-legal move generation.
No IO and no search - just the rules, kept independently testable.

## What's here

- **Bitboards** - one `u64` per piece kind in LERF order, with a parallel
  `[Option<Piece>; 64]` mailbox for O(1) square lookup.
- **Types** - `Square`, `Piece`, `Move`, `Color`, and `PieceType`; several wrap
  `NonMax*` so their `Option`s cost no extra space.
- **Move generation** - pseudo-legal moves for every piece. Castling is the lone
  exception, generated fully legally via `Board::is_attacked`.
- **make / unmake** - in-place move application with an `Undo` token, plus
  bulk-counted `perft` and `perft_divide`.
- **FEN** - single-pass, allocation-free `Board::from_fen`.

## Testing

```sh
cargo test -p lattice-board
cargo test -p lattice-board -- --ignored   # deeper perft (slow)
cargo bench -p lattice-board
```
