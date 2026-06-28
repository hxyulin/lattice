# Features

This page tracks the engine's currently implemented (and planned) features.

## Bitboards

A *bitboard* is a set of squares packed into a single 64-bit integer: one bit
per square, with bit `i` standing for square `i`. A chess board has exactly 64
squares, so a `u64` holds one bit for every square with none to spare.

Lattice uses **LERF** ordering (Little-Endian Rank-File): bit 0 is a1, bit 7 is
h1, bit 56 is a8, bit 63 is h8.

Representing a piece set this way turns board questions into single machine
instructions:

- **Union, intersection, difference** of square sets are `|`, `&`, and `& !`.
- **Population count** - how many squares are in the set - is `count_ones()`.
- **Iterating the set squares** pops the lowest set bit at a time using
  `trailing_zeros()` to read it and `x & (x - 1)` to clear it.

The engine keeps one bitboard per piece kind - twelve in all, six piece types
times two colors - so "where are the white knights?" is a direct lookup, and
"every white piece" is the union of six bitboards.

See the [Square Mapping] notes on the Chess Programming Wiki for background on
LERF and the alternatives.

[Square Mapping]: https://www.chessprogramming.org/Square_Mapping_Considerations
