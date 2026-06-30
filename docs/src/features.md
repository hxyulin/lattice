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

## Move generation

### Magic bitboards

Knights and kings have fixed attack patterns - a single table lookup per square.
Sliding pieces (bishop, rook, queen) are harder: their reach depends on which
squares are occupied, because the first blocker on a ray stops it. **Magic
bitboards** answer "where does this slider attack, given this occupancy?" in one
multiply, one shift, and one load.

The trick is a perfect hash. Only the squares *between* the slider and the board
edge can ever block a ray - the edge square itself cannot, since nothing sits
beyond it to be revealed - so each square has a *relevant-occupancy mask*. For a
given square there are only `2^popcount(mask)` distinct blocker arrangements that
matter. A **magic** is a multiplier chosen so that

```text
index = (occupancy & mask).wrapping_mul(magic) >> (64 - popcount(mask))
```

maps every one of those arrangements to a distinct slot in a small attack table.
The table is filled once with a slow reference slider (`slide`), which walks each
direction until it hits a blocker or the edge.

Lattice does not hard-code its magics; it **finds** them. A fixed-seed xorshift
PRNG proposes *sparse* candidates (three draws combined with bitwise AND, so few
bits are set - sparse magics hash cleaner and are found faster), and each candidate is
tested by filling the table and checking for a destructive collision: two
different attack sets colliding in the same slot. The first candidate that hashes
every occupancy cleanly is kept. A cheap pre-filter rejects most hopeless
candidates before the full fill by requiring the multiply to spread mask bits
into the top byte.

Because the seed and the square iteration order are fixed, the search is
deterministic - the same tables are produced on every run, on every machine. The
whole search runs once, lazily, and the UCI layer forces it during engine init
(`init_tables` on the `uci` command) so the one-time cost is paid up front rather
than mid-search.

See [Magic Bitboards] on the Chess Programming Wiki.

[Magic Bitboards]: https://www.chessprogramming.org/Magic_Bitboards

### Make / unmake and legality

The engine generates pseudo-legal moves and filters by **make then test**:
`make_move` applies a move and returns an undo token, `is_legal` checks the mover
is not left in check, and `unmake_move` restores the prior state from the token.
This keeps generation simple at the cost of one make/unmake per candidate. A
`make_null_move` / `unmake_null_move` pair does the same for the "pass" used by
null-move pruning (see below).

Correctness is held by **perft**: counting the leaf nodes of the move tree to a
fixed depth and comparing against known reference counts. Perft is the exhaustive
guard that the magic tables, special moves (castling, en passant, promotion), and
legality filtering are all exactly right.

## Board state

### Zobrist hashing

Every position carries a 64-bit **Zobrist hash**, an XOR of random keys: one per
(piece, square), plus keys for side to move, castling rights, and the en passant
file. Because XOR is its own inverse, the hash is maintained incrementally -
moving a piece XORs out its key on the old square and XORs in the key on the new
one - so a position's identity is recomputed in a handful of operations rather
than by scanning the board. This hash is the key into the transposition table.

### Incremental evaluation accumulator

The static evaluation (next section) is not recomputed from scratch at the
leaves. The board maintains running middlegame, endgame, and game-phase sums in
*accumulators*, updated in the same `put_piece` / `remove_piece` hooks that
maintain the Zobrist hash. Placing a piece adds its contribution; removing one
subtracts it. Evaluating a leaf is then just a phase clamp and a blend - no board
scan.

This is why the piece-square data lives in the board layer rather than the engine
layer: the board is the only place that observes every single piece placement, so
it is the only place the accumulator can be kept correct. It is also, not by
coincidence, the exact shape an NNUE accumulator would take - making this a
natural foundation for a future neural evaluation.

## Evaluation

### Material and piece-square tables (PeSTO)

The base evaluation is **material plus piece-square tables**, using Ronald
Friederich's PeSTO values. Material gives each piece a centipawn worth; the
piece-square table (PST) adds a positional bonus for *where* the piece stands - a
knight on a central outpost scores more than one in the corner, a pawn near
promotion more than one at home. Both are folded together into a single
per-(piece, square) delta baked at compile time, so the accumulator update is one
branch-free lookup.

The published PeSTO tables are laid out a8-first while this board is LERF, so the
build mirrors White through `sq ^ 56` and reads Black directly; the color sign
and the material value are baked into the same table entry.

### Tapered evaluation

A king belongs in the corner behind its pawns during the middlegame, and in the
centre during the endgame. A single static table cannot express both, so every
term carries **two** values - a middlegame (`mg`) and an endgame (`eg`) score -
and the final evaluation blends them by a **phase** scalar:

```text
score = (mg * phase + eg * (PHASE_MAX - phase)) / PHASE_MAX
```

The phase runs from 24 (all non-pawn material on the board) down to 0 (bare
kings), each piece type contributing a fixed weight, so the evaluation slides
*continuously* from opening to endgame as material comes off. Promotions can push
the running phase past the starting maximum, so it is clamped before the blend.
The blended score is White-relative and then flipped into the side-to-move's
frame, the convention negamax expects.

This is the engine's single biggest strength ceiling: the evaluation is currently
*only* material and PST. Mobility, king safety, pawn structure, and the rest are
not yet present (see [Planned](#planned)).

See [Tapered Eval] and [PeSTO] on the Chess Programming Wiki.

[Tapered Eval]: https://www.chessprogramming.org/Tapered_Eval
[PeSTO]: https://www.chessprogramming.org/PeSTO%27s_Evaluation_Function

## Search

The search finds the best move by looking ahead, scoring leaf positions with the
static evaluation, and backing those scores up the tree.

### Negamax with alpha-beta pruning

The tree walk is **negamax**, the single-function form of minimax that exploits
chess being zero-sum: a position's value to the side to move is the negation of
the best value the opponent can reach, so one routine handles both sides by
negating across each ply.

**Alpha-beta pruning** is what makes it fast. The search carries a window
`(alpha, beta)` - the score range still worth investigating. As soon as a move
proves it beats `beta` (the opponent already has a better alternative earlier in
the tree), the rest of that node's moves are skipped: searching them cannot change
the outcome. With good move ordering this prunes the overwhelming majority of the
tree without changing the result.

### Iterative deepening

Rather than searching directly to depth `N`, the engine searches to depth 1, then
2, then 3, and so on, keeping the best move from each completed iteration. This
sounds wasteful but is not: the shallow searches are cheap, and - crucially - each
iteration seeds the next with a strong best-move guess and a fully populated
transposition table, so the deeper search orders its moves far better and prunes
harder. It also makes the search **interruptible**: under a time or node budget,
the engine always has a complete, usable result from the last finished depth,
and a partial iteration aborted mid-flight is simply discarded.

The first iteration is guaranteed to finish (an `armed` gate ignores the stop
conditions until depth 1 banks a legal move), so even an instant time-out yields
a legal move.

### Transposition table

A position can be reached by many move orders, so the engine caches what each
search learned in a **transposition table** (TT), keyed by the Zobrist hash. Each
entry stores the depth searched, the best move found, a score, and a **bound**
flag describing what that score means - because alpha-beta returns truncated
values:

- **Exact** - the node was searched fully inside its window; the score is the
  true value.
- **Lower** - a beta cutoff; the true score is at least this (a fail-high).
- **Upper** - every move failed low; the true score is at most this.

A stored entry searched at least as deep as the current node can cut it off
outright when its bound permits (Exact always; Lower only if it still beats beta;
Upper only if it still falls below alpha). Even when the depth is too shallow to
cut, the stored **best move** is the single strongest move-ordering hint there is.

The table is a flat array of two-slot **buckets** indexed by the low hash bits.
Each bucket pairs a *depth-preferred* slot (kept when the incoming entry is not
deeper) with an *always-replace* slot (catches the shallow tail), so deep results
survive while recent shallow ones are still recorded. A per-search **generation**
counter ages entries: a new search bumps the generation, so this move's results
outrank the previous move's for replacement without wiping the table. Scores are
mate-distance corrected on the way in and out, so a "mate in N from here" stays
correct when the same entry is reused at a different depth. The full 64-bit key is
stored so an index collision (two positions sharing low bits) is rejected on
probe.

Deliberately simple for a single-threaded engine: plain non-atomic entries, no
lockless XOR trick, and quiescence nodes are not stored.

### Move ordering

Alpha-beta lives or dies by move ordering - the sooner the best move is tried,
the more gets pruned. Moves are scored and sorted by tiers:

1. **TT / PV move** - the best move from a previous search of this position (or
   the previous iteration at the root). Tried first; given a bonus far above every
   other tier.
2. **Captures, by MVV-LVA** - Most Valuable Victim, Least Valuable Attacker:
   `victim_value * 100 - attacker_value`, so grabbing a queen with a pawn sorts
   above grabbing a pawn with a queen. Winning and equal captures are generally
   strong and cheap to resolve.
3. **Killer moves** - up to two quiet moves per ply that caused a beta cutoff at a
   *sibling* node. A quiet refutation that worked one branch over often works here
   too, so killers are tried ahead of other quiets.
4. **History heuristic** - a butterfly table indexed `[side][from][to]`,
   accumulating a `depth^2` bonus each time a quiet move causes a cutoff
   (saturating at a cap). Quiets with no killer status fall back to this score, so
   moves that have historically been good across the whole search float up.

As a speed optimization, ordering is **skipped entirely below depth 2**: the
depth-1 frontier is the largest node layer but its children are leaves, so the
sort there would cost more than the sibling evaluations it saves.

### Principal variation search (PVS)

Once the first move is searched with the full `(alpha, beta)` window to establish
a baseline, every later move is **scouted** with a null (zero-width) window
`(alpha, alpha + 1)`. A null-window search only has to prove a move fails low,
which cuts off far sooner than a full search. If a scout unexpectedly beats alpha,
the move might be a new principal variation, so it is **re-searched** with the
full window for its exact value. With good ordering the first move usually is best,
so the cheap scouts dominate and the re-searches are rare. PVS runs at both the
root and interior nodes.

### Quiescence search

Stopping the search at a fixed depth invites the **horizon effect**: the engine
evaluates a position in the middle of a capture exchange and mistakes a hanging
piece for a won one. **Quiescence search** fixes this - at depth 0, instead of
evaluating immediately, the engine keeps searching *captures and promotions only*
until the position is quiet, then evaluates.

It is bounded three ways: a **stand-pat** cutoff (the static eval is a lower bound,
since the side to move can always decline to capture - if it already beats beta,
stop), the capture-only move set (every ply strictly removes material, so it
terminates), and a hard ply cap as a backstop.

**SEE pruning**: each candidate capture is screened by **Static Exchange
Evaluation**, which plays out the full capture/recapture sequence on one square
using the cheapest attacker each time and returns the net material swing. A
capture that loses material by SEE is almost never best, so it and its recapture
subtree are skipped - unless the side is in check (any capture might be a forced
escape) or the move is a promotion (SEE scores only the captured piece and misses
the queening gain).

### Pruning and reductions

Beyond alpha-beta, the search trims the tree with several heuristics, each
confined to where it is safe:

- **Null-move pruning (NMP)** - give the opponent a *free* move (pass); if a
  reduced search still beats beta even after handing them a tempo, the real
  position is so good it can be pruned. Gated on having non-pawn material
  (zugzwang, where passing is genuinely bad, essentially only happens in
  pawn endings), skipped in check, and mate scores from a reduced null search are
  capped so a false mate cannot leak out.
- **Reverse futility pruning (RFP)** - also called static null-move pruning. Near
  the leaves, if the static eval already clears beta by a depth-scaled margin, the
  side is far enough ahead that searching is almost certainly wasted, so fail high
  on the static score directly. Confined to shallow non-PV nodes, away from mate
  scores, and skipped in check.
- **Late move reductions (LMR)** - good ordering means moves past the first few
  are rarely best, so quiet non-killer moves beyond the leading prefix are
  *scouted shallower*. A reduced scout that beats alpha is verified at full depth
  before being trusted, so a reduction never silently drops a good move. The
  reduction amount comes from a precomputed `log(depth) * log(move_number)` table.
- **Late move pruning (LMP)** - at shallow non-PV nodes, once enough legal moves
  have been searched the remaining (worst-ordered) quiets are skipped *without
  being made at all*. The threshold grows quadratically with depth. Captures,
  promotions, and check evasions are exempt, and a mate guard keeps searching for
  an escape when the best score so far is a loss.

### Time management

Given a clock and increment, the engine allots roughly `remaining / 20 +
increment / 2` per move - a simple, robust budget. The search polls its stop
conditions cheaply (a node-count compare every node, the wall clock only every
2048 nodes to avoid a syscall per node) and unwinds fast once a limit fires.

## Tuning

A subset of search parameters - the RFP margin, the LMR base and divisor, the LMP
base, the history cap and bonus scale, and the killer ordering bonuses - are
exposed as integer UCI spin options so they can be tuned with **SPSA** under
OpenBench. The defaults reproduce the engine's built-in behavior byte-for-byte, so
an untuned search is unchanged (and the bench node signature is stable). Discrete
depth gates are kept as constants, where SPSA has little to work with.

## Planned

Roughly in Elo-per-effort order, the notable features not yet implemented:

- **Search:** aspiration windows, check extensions, SEE in move ordering,
  countermove and continuation history, classic futility pruning, razoring.
- **Evaluation:** mobility, king safety, pawn structure (passed / isolated /
  doubled), bishop pair, rook on open file - the bulk of the strength ceiling.
- **Longer term:** an NNUE evaluation, for which the incremental accumulator above
  is already the groundwork.
