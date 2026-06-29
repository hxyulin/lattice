//! Transposition table: maps a position's [`ZobristHash`] to what a previous
//! search learned, so a position reached by several move orders is searched once.
//!
//! # Notes
//!
//! Two payoffs: a stored score with `depth >= remaining depth` can cut this node
//! off (subject to its [`Bound`]), and even a too-shallow entry's best move is the
//! single best move-ordering hint.
//!
//! Layout is a flat `Vec` of 2-slot [`Bucket`]s indexed by the low hash bits; each
//! bucket pairs a depth-preferred slot with an always-replace slot, so deep
//! current entries survive while the shallow tail still gets recorded.
//!
//! Deliberate simplifications for a single-threaded engine: plain (non-atomic)
//! entries with no lockless XOR-key trick; quiescence is not probed; a cutoff
//! ignores the 50-move/repetition path (no repetition detection exists yet).

use lattice_board::{Move, ZobristHash};

use crate::search::MAX_PLY;
use crate::{MATE, Score};

/// What an entry's stored [`score`](Entry::score) means.
///
/// # Notes
///
/// Alpha-beta returns truncated values: a fail-high proves only a lower bound, a
/// fail-low only an upper bound, and only a node searched fully inside its window
/// yields the exact score. Probe logic differs per variant.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Bound {
    /// The true score - the node was searched inside `(alpha, beta)`.
    Exact,
    /// A lower bound: the true score is `>=` the stored one (a beta cutoff).
    Lower,
    /// An upper bound: the true score is `<=` the stored one (failed low).
    Upper,
}

/// One stored position.
///
/// # Performance
///
/// 16 bytes, so two fit in a [`Bucket`] within half a cache line. The full 64-bit
/// `key` is kept so a probe can reject an index collision (two positions whose low
/// hash bits coincide).
#[derive(Clone, Copy)]
pub struct Entry {
    key: u64,
    best: Option<Move>,
    /// Score with the mate-distance correction *applied* (node-relative); read it
    /// back through [`Entry::score`], which undoes the correction. `i16` holds it:
    /// `MATE` (30000) is well under `i16::MAX`.
    score: i16,
    depth: u8,
    bound: Bound,
    /// Search generation that wrote this. `0` marks an empty slot; real
    /// generations start at 1. Older generations are preferred for replacement.
    age: u8,
}

const _: () = assert!(size_of::<Entry>() == 16);
const _: () = assert!(size_of::<Bucket>() == 32);

impl Entry {
    /// A zeroed, empty slot (`gen == 0`).
    const EMPTY: Entry = Entry {
        key: 0,
        best: None,
        score: 0,
        depth: 0,
        bound: Bound::Exact,
        age: 0,
    };

    /// The best move recorded for this position, if any - the ordering hint.
    #[inline]
    #[must_use]
    pub fn best(&self) -> Option<Move> {
        self.best
    }

    /// Remaining depth this score was searched to. A cutoff is only valid when
    /// this is `>=` the current node's remaining depth.
    #[inline]
    #[must_use]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    /// What [`Self::score`] means at this node.
    #[inline]
    #[must_use]
    pub fn bound(&self) -> Bound {
        self.bound
    }

    /// The stored score, corrected back to be relative to `ply` plies from the
    /// root (undoing the node-relative storage of mate distances).
    #[inline]
    #[must_use]
    pub fn score(&self, ply: u32) -> Score {
        score_from_tt(self.score, ply)
    }
}

/// A 2-slot bucket: slot 0 is depth-preferred, slot 1 always-replace.
#[derive(Clone, Copy)]
struct Bucket {
    entries: [Entry; 2],
}

impl Bucket {
    const EMPTY: Bucket = Bucket {
        entries: [Entry::EMPTY; 2],
    };
}

/// A fixed-size transposition table. Owned by the engine driver and reused
/// across moves: [`new_search`](Self::new_search) ages the previous move's
/// entries, [`clear`](Self::clear) wipes it for a new game.
pub struct TranspositionTable {
    buckets: Vec<Bucket>,
    /// `nbuckets - 1`; `nbuckets` is a power of two so indexing is one mask.
    mask: u64,
    /// Current search generation, written into every stored entry. Bumped by
    /// [`Self::new_search`]; never `0` (that marks empty).
    generation: u8,
}

impl TranspositionTable {
    /// A table sized to about `megabytes` MB (rounded *down* to a power-of-two
    /// bucket count, so it never exceeds the budget). Minimum one bucket.
    #[must_use]
    pub fn new(megabytes: usize) -> Self {
        let mut tt = Self {
            buckets: Vec::new(),
            mask: 0,
            generation: 1,
        };
        tt.resize(megabytes);
        tt
    }

    /// Reallocate to about `megabytes` MB and clear. Used by `setoption Hash`.
    pub fn resize(&mut self, megabytes: usize) {
        let bytes = megabytes.max(1) * 1024 * 1024;
        let raw = (bytes / size_of::<Bucket>()).max(1);
        // Round DOWN to a power of two: `next_power_of_two` rounds up, which
        // could nearly double the requested memory.
        let up = raw.next_power_of_two();
        let nbuckets = if up > raw { up >> 1 } else { up }.max(1);
        self.buckets = vec![Bucket::EMPTY; nbuckets];
        self.mask = nbuckets as u64 - 1;
        self.generation = 1;
    }

    /// Wipe every entry (a new game). Keeps the current allocation.
    pub fn clear(&mut self) {
        self.buckets.iter_mut().for_each(|b| *b = Bucket::EMPTY);
        self.generation = 1;
    }

    /// Begin a new search: advance the generation so this move's stores outrank
    /// the previous move's for replacement. Skips `0` (the empty marker).
    pub fn new_search(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.generation = 1;
        }
    }

    /// Look up `key`. Returns a copy of the matching entry, or `None` on a miss.
    #[must_use]
    pub fn probe(&self, key: ZobristHash) -> Option<Entry> {
        let k = key.get();
        let bucket = &self.buckets[(k & self.mask) as usize];
        bucket
            .entries
            .iter()
            .find(|e| e.age != 0 && e.key == k)
            .copied()
    }

    /// Record a search result for `key`. `ply` (distance from the root) lets the
    /// stored score be mate-distance corrected. Picks the bucket slot by the
    /// 2-slot scheme: refresh a slot already holding this key, else take slot 0
    /// when the new entry is deeper or slot 0 is stale, otherwise slot 1.
    pub fn store(
        &mut self,
        key: ZobristHash,
        best: Option<Move>,
        score: Score,
        depth: u8,
        bound: Bound,
        ply: u32,
    ) {
        let k = key.get();
        let age = self.generation;
        let entry = Entry {
            key: k,
            best,
            score: score_to_tt(score, ply),
            depth,
            bound,
            age,
        };

        let bucket = &mut self.buckets[(k & self.mask) as usize];
        let [s0, s1] = &mut bucket.entries;
        let slot = if s0.key == k {
            s0
        } else if s1.key == k {
            s1
        } else if entry.depth >= s0.depth || s0.age != age {
            // Deeper than slot 0, or slot 0 is stale: overwrite it. The displaced
            // entry is not migrated; slot 1 catches the shallow tail on its own.
            s0
        } else {
            s1 // shallow current entry: into the always-replace slot
        };
        *slot = entry;
    }
}

#[cfg(test)]
impl TranspositionTable {
    /// Test-only: count live entries stored at exactly `depth`. Confirms
    /// quiescence writes its depth-0 entries - the main search never stores
    /// depth 0, since `negamax` returns to quiescence before its store path.
    pub(crate) fn count_at_depth(&self, depth: u8) -> usize {
        self.buckets
            .iter()
            .flat_map(|b| b.entries.iter())
            .filter(|e| e.age != 0 && e.depth == depth)
            .count()
    }
}

/// Threshold above which a score is a mate score (and needs ply correction):
/// a real mate is `MATE - dist` with `dist <= MAX_PLY`, so anything this large
/// is a mate, and no material eval reaches it.
const MATE_BOUND: Score = MATE - MAX_PLY as Score;

/// Make a root-relative score node-relative for storage. A mate `n` plies from
/// the root is `MATE - n`; adding `ply` rewrites it as "mate in `n - ply` from
/// *here*", which stays correct when the entry is reused at another depth.
fn score_to_tt(score: Score, ply: u32) -> i16 {
    let s = if score >= MATE_BOUND {
        score + ply as Score
    } else if score <= -MATE_BOUND {
        score - ply as Score
    } else {
        score
    };
    s as i16
}

/// Inverse of [`score_to_tt`]: turn a node-relative stored score back into a
/// root-relative one for the searcher at `ply`.
fn score_from_tt(score: i16, ply: u32) -> Score {
    let s = score as Score;
    if s >= MATE_BOUND {
        s - ply as Score
    } else if s <= -MATE_BOUND {
        s + ply as Score
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_board::{MoveFlag, Square};

    /// A move that is cheap to construct for use as a stored "best move".
    fn mv(from: &str, to: &str) -> Move {
        Move::new(
            Square::from_ascii(from.as_bytes()).unwrap(),
            Square::from_ascii(to.as_bytes()).unwrap(),
            MoveFlag::Quiet,
        )
    }

    /// Distinct hashes that share a bucket index (same low bits, different high
    /// bits) so collision behaviour can be exercised. Built from a real board's
    /// hash, then offset in the high bits only.
    fn keys() -> (ZobristHash, ZobristHash) {
        use lattice_board::Board;
        let a = Board::starting_position().zobrist();
        // Flip a high bit: same bucket (low bits unchanged), different key.
        let b = ZobristHash::from_raw(a.get() ^ (1 << 63));
        (a, b)
    }

    #[test]
    fn probe_after_store_round_trips() {
        let mut tt = TranspositionTable::new(1);
        let (k, _) = keys();
        let best = mv("e2", "e4");
        tt.store(k, Some(best), 123, 7, Bound::Exact, 0);
        let e = tt.probe(k).expect("just stored");
        assert_eq!(e.best(), Some(best));
        assert_eq!(e.score(0), 123);
        assert_eq!(e.depth(), 7);
        assert_eq!(e.bound(), Bound::Exact);
    }

    #[test]
    fn miss_returns_none() {
        let tt = TranspositionTable::new(1);
        let (k, _) = keys();
        assert!(tt.probe(k).is_none());
    }

    #[test]
    fn two_keys_share_a_bucket() {
        // Same index bits, different keys: both must coexist in the 2 slots.
        let mut tt = TranspositionTable::new(1);
        let (a, b) = keys();
        assert_eq!(
            a.get() & tt.mask,
            b.get() & tt.mask,
            "must collide by design"
        );
        tt.store(a, Some(mv("a2", "a3")), 10, 4, Bound::Exact, 0);
        tt.store(b, Some(mv("h2", "h3")), 20, 3, Bound::Lower, 0);
        assert_eq!(tt.probe(a).map(|e| e.depth()), Some(4));
        assert_eq!(tt.probe(b).map(|e| e.depth()), Some(3));
    }

    #[test]
    fn depth_preferred_slot_keeps_the_deeper_entry() {
        // Within one search (same generation), a deep entry in slot 0 is not
        // evicted by a shallower store of a *different* key - that goes to slot 1.
        let mut tt = TranspositionTable::new(1);
        let (a, b) = keys();
        tt.store(a, None, 0, 9, Bound::Exact, 0); // deep -> slot 0
        tt.store(b, None, 0, 2, Bound::Exact, 0); // shallow, diff key -> slot 1
        assert_eq!(
            tt.probe(a).map(|e| e.depth()),
            Some(9),
            "deep entry survives"
        );
        assert_eq!(tt.probe(b).map(|e| e.depth()), Some(2));
    }

    #[test]
    fn new_search_lets_a_shallow_entry_evict_a_stale_deep_one() {
        // A deep entry from a previous search (older generation) is replaceable
        // by a shallow entry in the new search.
        let mut tt = TranspositionTable::new(1);
        let (a, b) = keys();
        tt.store(a, None, 0, 9, Bound::Exact, 0); // deep, generation 1, slot 0
        tt.new_search(); // generation 2
        tt.store(b, None, 0, 1, Bound::Exact, 0); // shallow but newer -> slot 0
        assert_eq!(
            tt.probe(b).map(|e| e.depth()),
            Some(1),
            "stale deep entry yields"
        );
        assert!(tt.probe(a).is_none(), "the stale entry was overwritten");
    }

    #[test]
    fn clear_empties_the_table() {
        let mut tt = TranspositionTable::new(1);
        let (k, _) = keys();
        tt.store(k, None, 5, 3, Bound::Exact, 0);
        tt.clear();
        assert!(tt.probe(k).is_none());
    }

    #[test]
    fn mate_score_survives_a_ply_shift() {
        // Store a "mate in 3 from the root" (MATE - 3) discovered at ply 5, then
        // read it at ply 5: it must come back as MATE - 3, not drift.
        let mut tt = TranspositionTable::new(1);
        let (k, _) = keys();
        let mate_in_3 = MATE - 3;
        tt.store(k, None, mate_in_3, 4, Bound::Exact, 5);
        assert_eq!(tt.probe(k).unwrap().score(5), mate_in_3);
    }

    #[test]
    fn mate_correction_is_node_relative() {
        // The raw stored value is node-relative: a mate stored at one ply and
        // read at a different ply shifts by the ply delta (that is the point -
        // "mate in N from here" is depth-independent).
        let mut tt = TranspositionTable::new(1);
        let (k, _) = keys();
        tt.store(k, None, MATE - 6, 4, Bound::Exact, 6); // node-relative: MATE
        // Read as if this node were the root (ply 0): "mate right here".
        assert_eq!(tt.probe(k).unwrap().score(0), MATE);
    }
}
