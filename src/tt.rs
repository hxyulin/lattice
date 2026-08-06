//! Transposition table.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::Move;

/// Entries in the table. 2^20 entries at 16 bytes each is 16 MiB.
const BITS: u32 = 20;
const ENTRIES: usize = 1 << BITS;

/// What a stored score says about the true score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Bound {
    /// The search window was not truncated: the score is the true score.
    Exact = 0,
    /// A beta cutoff truncated the search: the true score is at least this.
    Lower = 1,
    /// No move beat alpha: the true score is at most this.
    Upper = 2,
}

impl Bound {
    fn from_bits(bits: u64) -> Self {
        match bits & 3 {
            0 => Self::Exact,
            1 => Self::Lower,
            _ => Self::Upper,
        }
    }
}

/// Why a probe found what it found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Miss {
    /// The slot held this position.
    Hit,
    /// The slot was never written, so nothing was displaced.
    Empty,
    /// The slot held a different position, which this one would evict.
    Collision,
}

/// A probed entry.
#[derive(Debug, Clone, Copy)]
pub struct Entry {
    /// Score as stored, with mate distances relative to the entry's own node.
    pub score: i32,
    /// Best move found, if one was recorded.
    pub best_move: Option<Move>,
    /// Remaining depth this entry was searched to.
    pub depth: u32,
    /// What the score says about the true score.
    pub bound: Bound,
}

/// A lockless fixed-size transposition table.
///
/// Each slot holds `key ^ data` beside `data`, so a read torn against a
/// concurrent write recovers a key that does not match and is discarded. A
/// stale or colliding entry can therefore only ever be a miss, never a valid
/// entry carrying another position's move.
#[derive(Debug)]
pub struct TranspositionTable {
    slots: Vec<(AtomicU64, AtomicU64)>,
    generation: AtomicU64,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    /// Builds an empty table of the fixed default size.
    pub fn new() -> Self {
        Self {
            slots: (0..ENTRIES)
                .map(|_| (AtomicU64::new(0), AtomicU64::new(0)))
                .collect(),
            generation: AtomicU64::new(0),
        }
    }

    /// Marks every entry as belonging to an older search, so the next one
    /// replaces them freely. Cheaper than zeroing 16 MiB.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Empties the table. Used between games, where carrying scores over is
    /// not merely stale but wrong.
    pub fn clear(&self) {
        for slot in &self.slots {
            slot.0.store(0, Ordering::Relaxed);
            slot.1.store(0, Ordering::Relaxed);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    fn index(&self, key: u64) -> usize {
        // The high bits: the low ones are the least mixed by `mix`.
        (key >> (64 - BITS)) as usize
    }

    /// Returns the entry stored for `key`, if the slot still holds it.
    pub fn probe(&self, key: u64) -> Option<Entry> {
        self.probe_kind(key).0
    }

    /// `probe`, plus why a miss missed.
    ///
    /// An empty slot means the table has room; an occupied slot holding
    /// another position means it does not, and that entries are evicting each
    /// other. The two want opposite responses, so a bare hit rate cannot tell
    /// whether the table is too small.
    pub fn probe_kind(&self, key: u64) -> (Option<Entry>, Miss) {
        let slot = &self.slots[self.index(key)];
        let stored_key = slot.0.load(Ordering::Relaxed);
        let data = slot.1.load(Ordering::Relaxed);
        if data == 0 {
            return (None, Miss::Empty);
        }
        if stored_key ^ data != key {
            return (None, Miss::Collision);
        }
        let entry = Entry {
            score: (data & 0xffff) as u16 as i16 as i32,
            best_move: Move::from_bits(((data >> 16) & 0xffff) as u16),
            depth: ((data >> 32) & 0xff) as u32,
            bound: Bound::from_bits(data >> 40),
        };
        (Some(entry), Miss::Hit)
    }

    /// Records a search result for `key`.
    ///
    /// Keeps a deeper entry for the same position rather than replacing it: a
    /// shallow iteration of a new search would otherwise discard the deep
    /// result the previous one left there, which is the entry the new search
    /// most wants. A different position takes the slot regardless of depth,
    /// and an entry from an older search loses ties.
    pub fn store(&self, key: u64, score: i32, best_move: Option<Move>, depth: u32, bound: Bound) {
        let slot = &self.slots[self.index(key)];
        let generation = self.generation.load(Ordering::Relaxed);
        let existing = slot.1.load(Ordering::Relaxed);
        let same_position = existing != 0 && slot.0.load(Ordering::Relaxed) ^ existing == key;
        if same_position && (existing >> 32) & 0xff > u64::from(depth) {
            return;
        }
        let data = (score as i16 as u16 as u64)
            | ((best_move.map_or(0, Move::to_bits) as u64) << 16)
            | ((depth.min(0xff) as u64) << 32)
            | ((bound as u64) << 40)
            | ((generation & 0xff) << 42);
        // Non-zero so `probe` can treat an all-zero slot as empty. `data` is
        // only zero for an Exact score of 0 with no move at depth 0, which
        // carries nothing worth keeping anyway.
        if data == 0 {
            return;
        }
        slot.0.store(key ^ data, Ordering::Relaxed);
        slot.1.store(data, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MoveType, Square};

    fn mv(from: u8, to: u8) -> Move {
        Move::new(
            Square::new_unchecked(from),
            Square::new_unchecked(to),
            MoveType::Quiet,
        )
    }

    #[test]
    fn roundtrips_every_field() {
        let tt = TranspositionTable::new();
        let m = mv(12, 28);
        tt.store(0xdead_beef_1234_5678, -1234, Some(m), 7, Bound::Upper);
        let entry = tt.probe(0xdead_beef_1234_5678).expect("just stored");
        assert_eq!(entry.score, -1234);
        assert_eq!(entry.best_move, Some(m));
        assert_eq!(entry.depth, 7);
        assert_eq!(entry.bound, Bound::Upper);
    }

    #[test]
    fn a_different_key_in_the_same_slot_misses() {
        let tt = TranspositionTable::new();
        // Same index (top BITS bits), different key.
        let key = 0xffff_0000_0000_0000;
        let other = key | 1;
        tt.store(key, 42, Some(mv(1, 2)), 4, Bound::Exact);
        assert_eq!(tt.index(key), tt.index(other));
        assert!(tt.probe(other).is_none(), "key check must reject");
        assert!(tt.probe(key).is_some());
    }

    // catches: a miss reason that cannot tell an unused slot from an evicting
    // one. Both return None, so only the reason distinguishes "the table has
    // room" from "the table is thrashing", which want opposite responses.
    #[test]
    fn a_miss_reports_whether_the_slot_was_empty_or_taken() {
        let tt = TranspositionTable::new();
        let key = 0xffff_0000_0000_0000;
        let other = key | 1;
        assert_eq!(tt.index(key), tt.index(other));
        assert_eq!(tt.probe_kind(key).1, Miss::Empty);
        tt.store(key, 42, Some(mv(1, 2)), 4, Bound::Exact);
        assert_eq!(tt.probe_kind(key).1, Miss::Hit);
        assert_eq!(
            tt.probe_kind(other).1,
            Miss::Collision,
            "an occupied slot holding another position is not an empty one"
        );
        tt.clear();
        assert_eq!(tt.probe_kind(key).1, Miss::Empty);
    }

    #[test]
    fn empty_slots_and_cleared_tables_miss() {
        let tt = TranspositionTable::new();
        assert!(tt.probe(0x1234).is_none());
        tt.store(0x1234, 5, Some(mv(3, 4)), 2, Bound::Exact);
        assert!(tt.probe(0x1234).is_some());
        tt.clear();
        assert!(tt.probe(0x1234).is_none(), "clear must empty the table");
    }

    #[test]
    fn a_deeper_entry_survives_a_shallower_store() {
        let tt = TranspositionTable::new();
        let key = 0x1234_5678_9abc_def0;
        tt.store(key, 100, Some(mv(1, 2)), 9, Bound::Exact);
        tt.store(key, 200, Some(mv(3, 4)), 2, Bound::Exact);
        let entry = tt.probe(key).expect("deep entry must remain");
        assert_eq!(entry.depth, 9);
        assert_eq!(entry.score, 100);
        // A new search must not discard it either: its shallow first
        // iterations would otherwise throw away the result they want most.
        tt.new_search();
        tt.store(key, 200, Some(mv(3, 4)), 2, Bound::Exact);
        assert_eq!(tt.probe(key).expect("kept").depth, 9);
        // Reaching the same depth again does replace, so bounds can tighten
        // and a re-searched score can correct a stale one.
        tt.store(key, 300, Some(mv(5, 6)), 9, Bound::Exact);
        assert_eq!(tt.probe(key).expect("replaced").score, 300);
    }

    #[test]
    fn a_different_position_takes_the_slot_regardless_of_depth() {
        // Depth preference applies to the same position. A colliding key must
        // not be locked out by a deep entry it can never match.
        let tt = TranspositionTable::new();
        let key = 0xffff_0000_0000_0000;
        let other = key | 0xff;
        assert_eq!(tt.index(key), tt.index(other));
        tt.store(key, 100, Some(mv(1, 2)), 9, Bound::Exact);
        tt.store(other, 200, Some(mv(3, 4)), 1, Bound::Exact);
        assert!(tt.probe(key).is_none(), "evicted by the colliding key");
        assert_eq!(tt.probe(other).expect("stored").score, 200);
    }

    #[test]
    fn equal_depth_replaces_so_bounds_can_tighten() {
        let tt = TranspositionTable::new();
        let key = 0xabcd_ef01_2345_6789;
        tt.store(key, 10, Some(mv(1, 2)), 5, Bound::Lower);
        tt.store(key, 20, Some(mv(3, 4)), 5, Bound::Exact);
        let entry = tt.probe(key).expect("stored");
        assert_eq!(entry.bound, Bound::Exact);
        assert_eq!(entry.score, 20);
    }

    #[test]
    fn negative_and_extreme_scores_survive_the_round_trip() {
        let tt = TranspositionTable::new();
        for score in [-30_000, -1, 0, 1, 30_000] {
            let key = 0x5555_0000_0000_0000 ^ ((score as i64 as u64) << 8);
            tt.store(key, score, None, 3, Bound::Exact);
            assert_eq!(tt.probe(key).expect("stored").score, score, "{score}");
        }
    }
}
