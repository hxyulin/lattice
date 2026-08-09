//! Compact placement aggregation and deterministic train/validation splitting.

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use super::data::{RECORD_BYTES, decode_placement};
use super::features::{FeatureTerm, PARAMETER_COUNT, extract};

#[derive(Debug)]
pub(crate) struct DatasetError(String);

impl fmt::Display for DatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DatasetError {}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LabelCounts {
    pub(crate) losses: u32,
    pub(crate) draws: u32,
    pub(crate) wins: u32,
}

impl LabelCounts {
    pub(crate) fn weight(self) -> u64 {
        u64::from(self.losses) + u64::from(self.draws) + u64::from(self.wins)
    }

    pub(crate) fn soft_target(self) -> f64 {
        (f64::from(self.wins) + 0.5 * f64::from(self.draws)) / self.weight() as f64
    }

    fn add(&mut self, result: u8) {
        match result {
            0 => self.losses += 1,
            1 => self.draws += 1,
            2 => self.wins += 1,
            _ => unreachable!("result was validated while loading"),
        }
    }

    fn conflicts(self) -> bool {
        usize::from(self.losses != 0) + usize::from(self.draws != 0) + usize::from(self.wins != 0)
            > 1
    }
}

/// One contiguous sparse partition.
#[derive(Debug, Default)]
pub(crate) struct Partition {
    pub(crate) offsets: Vec<u32>,
    pub(crate) entries: Vec<FeatureTerm>,
    pub(crate) labels: Vec<LabelCounts>,
    pub(crate) total_weight: u64,
}

impl Partition {
    pub(crate) fn len(&self) -> usize {
        self.labels.len()
    }

    pub(crate) fn terms(&self, sample: usize) -> &[FeatureTerm] {
        &self.entries[self.offsets[sample] as usize..self.offsets[sample + 1] as usize]
    }

    fn push(&mut self, terms: &[FeatureTerm], labels: LabelCounts) -> Result<(), DatasetError> {
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.entries.extend_from_slice(terms);
        self.offsets
            .push(u32::try_from(self.entries.len()).map_err(|_| {
                DatasetError("sparse feature storage exceeded 2^32 entries".to_owned())
            })?);
        self.total_weight += labels.weight();
        self.labels.push(labels);
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct DatasetStats {
    pub(crate) bytes: u64,
    pub(crate) records: u64,
    pub(crate) wdl: [u64; 3],
    pub(crate) unique_placements: usize,
    pub(crate) conflicting_placements: usize,
    pub(crate) train_placements: usize,
    pub(crate) validation_placements: usize,
    pub(crate) train_weight: u64,
    pub(crate) validation_weight: u64,
    pub(crate) feature_entries: usize,
    pub(crate) unsupported_parameters: Vec<usize>,
    pub(crate) max_baseline_parity_error: f64,
}

#[derive(Debug)]
pub(crate) struct Dataset {
    pub(crate) train: Partition,
    pub(crate) validation: Partition,
    pub(crate) stats: DatasetStats,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CompactRecord {
    placement: [u8; 24],
    result: u8,
}

/// Loads, deduplicates, partitions, decodes and sparsifies input shards.
pub(crate) fn load(
    paths: &[PathBuf],
    validation_fraction: f64,
    seed: u64,
) -> Result<Dataset, DatasetError> {
    if paths.is_empty() {
        return Err(DatasetError(
            "at least one shard path is required".to_owned(),
        ));
    }
    if !(0.0 < validation_fraction && validation_fraction < 1.0) {
        return Err(DatasetError(
            "validation fraction must be between 0 and 1".to_owned(),
        ));
    }

    let mut compact = Vec::new();
    let mut stats = DatasetStats::default();
    for path in paths {
        load_compact(path, &mut compact, &mut stats)?;
    }
    compact.sort_unstable();

    crate::movegen::init();
    let baseline = crate::eval::tuning_parameters();
    let mut train = Partition::default();
    let mut validation = Partition::default();
    let mut support = [0u64; PARAMETER_COUNT];
    let threshold = (validation_fraction * 10_000.0).round() as u64;

    let mut index = 0;
    while index < compact.len() {
        let placement = compact[index].placement;
        let mut labels = LabelCounts::default();
        let mut end = index;
        while end < compact.len() && compact[end].placement == placement {
            labels.add(compact[end].result);
            end += 1;
        }
        stats.unique_placements += 1;
        stats.conflicting_placements += usize::from(labels.conflicts());

        let board = decode_placement(&placement).map_err(|error| {
            DatasetError(format!(
                "invalid placement at sorted sample {}: {error}",
                stats.unique_placements - 1
            ))
        })?;
        let terms = extract(&board);
        let parity_error = (super::features::score(&baseline, &terms)
            - f64::from(crate::eval::evaluate(&board)))
        .abs();
        if parity_error >= 1.000_1 {
            return Err(DatasetError(format!(
                "sparse/runtime evaluation mismatch at placement {}: {parity_error:.6} cp",
                stats.unique_placements - 1
            )));
        }
        stats.max_baseline_parity_error = stats.max_baseline_parity_error.max(parity_error);
        let is_validation = stable_hash(&placement, seed) % 10_000 < threshold;
        if !is_validation {
            for term in &terms {
                support[term.index as usize] += labels.weight();
            }
        }
        let target = if is_validation {
            &mut validation
        } else {
            &mut train
        };
        target.push(&terms, labels)?;
        index = end;
    }

    if train.len() == 0 || validation.len() == 0 {
        return Err(DatasetError(
            "deterministic split produced an empty partition".to_owned(),
        ));
    }
    stats.train_placements = train.len();
    stats.validation_placements = validation.len();
    stats.train_weight = train.total_weight;
    stats.validation_weight = validation.total_weight;
    stats.feature_entries = train.entries.len() + validation.entries.len();
    stats.unsupported_parameters = support
        .into_iter()
        .enumerate()
        .filter_map(|(index, weight)| (weight == 0).then_some(index))
        .collect();

    if train.total_weight + validation.total_weight != stats.records {
        return Err(DatasetError(format!(
            "aggregation weight {} does not match input record count {}",
            train.total_weight + validation.total_weight,
            stats.records
        )));
    }
    Ok(Dataset {
        train,
        validation,
        stats,
    })
}

fn load_compact(
    path: &Path,
    compact: &mut Vec<CompactRecord>,
    stats: &mut DatasetStats,
) -> Result<(), DatasetError> {
    let bytes = fs::read(path)
        .map_err(|error| DatasetError(format!("reading {}: {error}", path.display())))?;
    if !bytes.len().is_multiple_of(RECORD_BYTES) {
        return Err(DatasetError(format!(
            "{}: {} bytes is not a whole number of {RECORD_BYTES}-byte records",
            path.display(),
            bytes.len()
        )));
    }
    stats.bytes += bytes.len() as u64;
    compact.reserve(bytes.len() / RECORD_BYTES);
    for record in bytes.chunks_exact(RECORD_BYTES) {
        let result = record[26];
        if result > 2 {
            return Err(DatasetError(format!(
                "{}: invalid result {result} at record {}",
                path.display(),
                stats.records
            )));
        }
        let mut placement = [0; 24];
        placement.copy_from_slice(&record[..24]);
        compact.push(CompactRecord { placement, result });
        stats.wdl[result as usize] += 1;
        stats.records += 1;
    }
    Ok(())
}

fn stable_hash(placement: &[u8; 24], seed: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64 ^ seed;
    for byte in placement {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_counts_preserve_occurrence_target() {
        let labels = LabelCounts {
            losses: 2,
            draws: 3,
            wins: 5,
        };
        assert_eq!(labels.weight(), 10);
        assert_eq!(labels.soft_target(), 0.65);
        assert!(labels.conflicts());
    }

    #[test]
    fn stable_split_hash_depends_on_seed_but_not_process_state() {
        let placement = [42; 24];
        assert_eq!(stable_hash(&placement, 7), stable_hash(&placement, 7));
        assert_ne!(stable_hash(&placement, 7), stable_hash(&placement, 8));
    }
}
