//! Texel probability calibration and deterministic sparse full-batch Adam.

use std::fmt;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use super::dataset::{DatasetStats, LabelCounts, Partition, load};
use super::features::{PARAMETER_COUNT, score};
use super::output::write_artifacts;

/// Complete configuration for one tuning run.
#[derive(Debug, Clone)]
pub struct TuneConfig {
    /// Nyquist BulletFormat shard paths.
    pub shards: Vec<PathBuf>,
    /// Directory receiving generated weights and the training report.
    pub output: PathBuf,
    /// Maximum Adam epochs.
    pub epochs: usize,
    /// Adam learning rate in centipawns.
    pub learning_rate: f64,
    /// Fraction of unique placements assigned to validation.
    pub validation_fraction: f64,
    /// Validation checks without improvement before early stopping.
    pub patience: usize,
    /// Stable partition seed.
    pub seed: u64,
    /// Worker threads used for loss and gradient accumulation.
    pub threads: usize,
}

/// Final metrics and artifact location from a successful tune.
#[derive(Debug, Clone)]
pub struct TuneSummary {
    /// Calibrated, frozen logistic scale.
    pub k: f64,
    /// Epoch containing the best continuous validation checkpoint.
    pub best_epoch: usize,
    /// Baseline validation weighted MSE.
    pub baseline_validation_loss: f64,
    /// Best continuous validation weighted MSE.
    pub best_validation_loss: f64,
    /// Rounded integer validation weighted MSE.
    pub rounded_validation_loss: f64,
    /// Unique canonical placements loaded.
    pub unique_placements: usize,
    /// Total occurrence weight loaded.
    pub records: u64,
    /// Directory containing `tuned_weights.rs` and `training-report.txt`.
    pub output: PathBuf,
}

/// A data, numerical, configuration, or output failure during tuning.
#[derive(Debug)]
pub struct TuneError(String);

impl fmt::Display for TuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TuneError {}

impl From<std::io::Error> for TuneError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

/// Runs the full load, calibration, optimization, rounding and output pipeline.
pub fn run(config: &TuneConfig) -> Result<TuneSummary, TuneError> {
    validate_config(config)?;
    let started = Instant::now();
    eprintln!("tune: loading {} shard(s)", config.shards.len());
    let dataset = load(&config.shards, config.validation_fraction, config.seed)
        .map_err(|error| TuneError(error.to_string()))?;
    report_dataset(&dataset.stats);

    let baseline = crate::eval::tuning_parameters();
    eprintln!("tune: calibrating logistic K");
    let train_scores = partition_scores(&dataset.train, &baseline);
    let k = calibrate_k(
        &train_scores,
        &dataset.train.labels,
        dataset.train.total_weight,
    );
    let baseline_train = labelled_score_loss(
        &train_scores,
        &dataset.train.labels,
        dataset.train.total_weight,
        k,
    );
    let baseline_validation = loss(&dataset.validation, &baseline, k, config.threads);
    eprintln!(
        "tune: K={k:.8} baseline train={baseline_train:.10} validation={baseline_validation:.10}"
    );

    let mut parameters = baseline;
    let mut first_moment = [0.0; PARAMETER_COUNT];
    let mut second_moment = [0.0; PARAMETER_COUNT];
    let mut best_parameters = parameters;
    let mut best_validation = baseline_validation;
    let mut best_epoch = 0usize;
    let mut stale = 0usize;
    let unsupported: std::collections::HashSet<_> = dataset
        .stats
        .unsupported_parameters
        .iter()
        .copied()
        .collect();

    for epoch in 1..=config.epochs {
        let accumulation = accumulate(&dataset.train, &parameters, k, config.threads, true);
        let train_loss_before_update = accumulation.loss / dataset.train.total_weight as f64;
        let beta1_power = 0.9f64.powi(epoch as i32);
        let beta2_power = 0.999f64.powi(epoch as i32);
        for index in 0..PARAMETER_COUNT {
            if unsupported.contains(&index) {
                continue;
            }
            let gradient = accumulation.gradient[index] / dataset.train.total_weight as f64;
            first_moment[index] = 0.9 * first_moment[index] + 0.1 * gradient;
            second_moment[index] = 0.999 * second_moment[index] + 0.001 * gradient * gradient;
            let corrected_first = first_moment[index] / (1.0 - beta1_power);
            let corrected_second = second_moment[index] / (1.0 - beta2_power);
            parameters[index] -=
                config.learning_rate * corrected_first / (corrected_second.sqrt() + 1.0e-8);
            if !parameters[index].is_finite() {
                return Err(TuneError(format!(
                    "parameter {index} became non-finite at epoch {epoch}"
                )));
            }
        }

        let validation_loss = loss(&dataset.validation, &parameters, k, config.threads);
        if !validation_loss.is_finite() || !train_loss_before_update.is_finite() {
            return Err(TuneError(format!("non-finite loss at epoch {epoch}")));
        }
        let improved = validation_loss + 1.0e-12 < best_validation;
        if improved {
            best_validation = validation_loss;
            best_parameters = parameters;
            best_epoch = epoch;
            stale = 0;
        } else {
            stale += 1;
        }
        if epoch == 1 || epoch % 5 == 0 || improved && epoch <= 10 {
            eprintln!(
                "tune: epoch {epoch:>3} train(pre)={train_loss_before_update:.10} validation={validation_loss:.10} best={best_validation:.10}"
            );
        }
        if stale >= config.patience {
            eprintln!("tune: early stop after {stale} stale validation epochs");
            break;
        }
    }

    if best_epoch == 0 || best_validation >= baseline_validation {
        return Err(TuneError(format!(
            "optimization did not improve validation loss ({baseline_validation:.10} -> {best_validation:.10})"
        )));
    }

    let best_train = loss(&dataset.train, &best_parameters, k, config.threads);
    let rounded: [i32; PARAMETER_COUNT] = std::array::from_fn(|index| {
        best_parameters[index]
            .round()
            .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    });
    let rounded_f64: [f64; PARAMETER_COUNT] =
        std::array::from_fn(|index| f64::from(rounded[index]));
    let rounded_train = loss(&dataset.train, &rounded_f64, k, config.threads);
    let rounded_validation = loss(&dataset.validation, &rounded_f64, k, config.threads);
    if rounded_validation > best_validation * 1.001 {
        return Err(TuneError(format!(
            "integer rounding regressed validation loss by more than 0.1% ({best_validation:.10} -> {rounded_validation:.10})"
        )));
    }
    if rounded_validation >= baseline_validation {
        return Err(TuneError(format!(
            "rounded weights do not improve baseline validation loss ({baseline_validation:.10} -> {rounded_validation:.10})"
        )));
    }

    let elapsed = started.elapsed();
    let report = training_report(
        config,
        &dataset.stats,
        k,
        baseline_train,
        baseline_validation,
        best_epoch,
        best_train,
        best_validation,
        rounded_train,
        rounded_validation,
        elapsed.as_secs_f64(),
    );
    write_artifacts(&config.output, &rounded, &report)
        .map_err(|error| TuneError(format!("writing {}: {error}", config.output.display())))?;
    eprintln!(
        "tune: complete epoch={best_epoch} validation {baseline_validation:.10} -> {rounded_validation:.10} in {:.1}s",
        elapsed.as_secs_f64()
    );

    Ok(TuneSummary {
        k,
        best_epoch,
        baseline_validation_loss: baseline_validation,
        best_validation_loss: best_validation,
        rounded_validation_loss: rounded_validation,
        unique_placements: dataset.stats.unique_placements,
        records: dataset.stats.records,
        output: config.output.clone(),
    })
}

fn validate_config(config: &TuneConfig) -> Result<(), TuneError> {
    if config.epochs == 0 {
        return Err(TuneError("epochs must be greater than zero".to_owned()));
    }
    if !(config.learning_rate.is_finite() && config.learning_rate > 0.0) {
        return Err(TuneError(
            "learning rate must be finite and positive".to_owned(),
        ));
    }
    if config.patience == 0 {
        return Err(TuneError("patience must be greater than zero".to_owned()));
    }
    if config.threads == 0 {
        return Err(TuneError("threads must be greater than zero".to_owned()));
    }
    Ok(())
}

fn report_dataset(stats: &DatasetStats) {
    eprintln!(
        "tune: records={} unique={} conflicts={} train={}/{} validation={}/{} features={} unsupported={} parity_max={:.6}cp",
        stats.records,
        stats.unique_placements,
        stats.conflicting_placements,
        stats.train_placements,
        stats.train_weight,
        stats.validation_placements,
        stats.validation_weight,
        stats.feature_entries,
        stats.unsupported_parameters.len(),
        stats.max_baseline_parity_error
    );
}

fn partition_scores(partition: &Partition, parameters: &[f64]) -> Vec<f64> {
    (0..partition.len())
        .map(|sample| score(parameters, partition.terms(sample)))
        .collect()
}

fn calibrate_k(scores: &[f64], labels: &[LabelCounts], total_weight: u64) -> f64 {
    let mut best_k = 0.05;
    let mut best_loss = f64::INFINITY;
    for step in 1..=60 {
        let k = step as f64 * 0.05;
        let loss = labelled_score_loss(scores, labels, total_weight, k);
        if loss < best_loss {
            best_loss = loss;
            best_k = k;
        }
    }
    let mut left = (best_k - 0.05).max(0.001);
    let mut right = best_k + 0.05;
    const RATIO: f64 = 0.618_033_988_749_894_9;
    let mut c = right - RATIO * (right - left);
    let mut d = left + RATIO * (right - left);
    let mut fc = labelled_score_loss(scores, labels, total_weight, c);
    let mut fd = labelled_score_loss(scores, labels, total_weight, d);
    for _ in 0..36 {
        if fc < fd {
            right = d;
            d = c;
            fd = fc;
            c = right - RATIO * (right - left);
            fc = labelled_score_loss(scores, labels, total_weight, c);
        } else {
            left = c;
            c = d;
            fc = fd;
            d = left + RATIO * (right - left);
            fd = labelled_score_loss(scores, labels, total_weight, d);
        }
    }
    (left + right) * 0.5
}

fn labelled_score_loss(scores: &[f64], labels: &[LabelCounts], total_weight: u64, k: f64) -> f64 {
    scores
        .iter()
        .zip(labels)
        .map(|(&score, &labels)| sample_loss(prediction(score, k), labels))
        .sum::<f64>()
        / total_weight as f64
}

fn prediction(score: f64, k: f64) -> f64 {
    let x = std::f64::consts::LN_10 * k * score / 400.0;
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

fn sample_loss(prediction: f64, labels: LabelCounts) -> f64 {
    f64::from(labels.losses) * prediction * prediction
        + f64::from(labels.draws) * (prediction - 0.5).powi(2)
        + f64::from(labels.wins) * (prediction - 1.0).powi(2)
}

struct Accumulation {
    loss: f64,
    gradient: Vec<f64>,
}

fn loss(partition: &Partition, parameters: &[f64], k: f64, threads: usize) -> f64 {
    accumulate(partition, parameters, k, threads, false).loss / partition.total_weight as f64
}

fn accumulate(
    partition: &Partition,
    parameters: &[f64],
    k: f64,
    threads: usize,
    with_gradient: bool,
) -> Accumulation {
    let chunk_count = partition.len().clamp(1, 64);
    let chunk_size = partition.len().div_ceil(chunk_count);
    let next = AtomicUsize::new(0);
    let slots: Vec<Mutex<Option<Accumulation>>> =
        (0..chunk_count).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        for _ in 0..threads.min(chunk_count) {
            scope.spawn(|| {
                loop {
                    let chunk = next.fetch_add(1, Ordering::Relaxed);
                    if chunk >= chunk_count {
                        break;
                    }
                    let start = chunk * chunk_size;
                    let end = (start + chunk_size).min(partition.len());
                    let accumulation =
                        accumulate_range(partition, parameters, k, start, end, with_gradient);
                    *slots[chunk].lock().expect("chunk result mutex poisoned") = Some(accumulation);
                }
            });
        }
    });
    let mut total = Accumulation {
        loss: 0.0,
        gradient: vec![0.0; PARAMETER_COUNT],
    };
    for slot in slots {
        let partial = slot
            .into_inner()
            .expect("chunk result mutex poisoned")
            .expect("every chunk is processed");
        total.loss += partial.loss;
        if with_gradient {
            for (total, partial) in total.gradient.iter_mut().zip(partial.gradient) {
                *total += partial;
            }
        }
    }
    total
}

fn accumulate_range(
    partition: &Partition,
    parameters: &[f64],
    k: f64,
    start: usize,
    end: usize,
    with_gradient: bool,
) -> Accumulation {
    let mut out = Accumulation {
        loss: 0.0,
        gradient: vec![0.0; PARAMETER_COUNT],
    };
    let logistic_scale = std::f64::consts::LN_10 * k / 400.0;
    for sample in start..end {
        let terms = partition.terms(sample);
        let labels = partition.labels[sample];
        let predicted = prediction(score(parameters, terms), k);
        out.loss += sample_loss(predicted, labels);
        if with_gradient {
            let weight = labels.weight() as f64;
            let target = labels.soft_target();
            let derivative = 2.0
                * weight
                * (predicted - target)
                * logistic_scale
                * predicted
                * (1.0 - predicted);
            for term in terms {
                out.gradient[term.index as usize] += derivative * f64::from(term.coefficient);
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn training_report(
    config: &TuneConfig,
    stats: &DatasetStats,
    k: f64,
    baseline_train: f64,
    baseline_validation: f64,
    best_epoch: usize,
    best_train: f64,
    best_validation: f64,
    rounded_train: f64,
    rounded_validation: f64,
    elapsed_seconds: f64,
) -> String {
    let shards = config
        .shards
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Lattice Texel tuning report\n\
         shards: {shards}\n\
         bytes: {}\nrecords: {}\nwdl: {},{},{}\n\
         unique_placements: {}\nconflicting_placements: {}\n\
         train: {} placements / {} occurrences\n\
         validation: {} placements / {} occurrences\n\
         sparse_feature_entries: {}\nunsupported_parameters: {:?}\nmax_baseline_parity_error_cp: {:.9}\n\
         seed: {}\nvalidation_fraction: {}\nthreads: {}\n\
         max_epochs: {}\nlearning_rate: {}\npatience: {}\n\
         logistic_k: {k:.12}\nbest_epoch: {best_epoch}\n\
         baseline_train_loss: {baseline_train:.12}\n\
         baseline_validation_loss: {baseline_validation:.12}\n\
         best_continuous_train_loss: {best_train:.12}\n\
         best_continuous_validation_loss: {best_validation:.12}\n\
         rounded_train_loss: {rounded_train:.12}\n\
         rounded_validation_loss: {rounded_validation:.12}\n\
         elapsed_seconds: {elapsed_seconds:.3}\n",
        stats.bytes,
        stats.records,
        stats.wdl[0],
        stats.wdl[1],
        stats.wdl[2],
        stats.unique_placements,
        stats.conflicting_placements,
        stats.train_placements,
        stats.train_weight,
        stats.validation_placements,
        stats.validation_weight,
        stats.feature_entries,
        stats.unsupported_parameters,
        stats.max_baseline_parity_error,
        config.seed,
        config.validation_fraction,
        config.threads,
        config.epochs,
        config.learning_rate,
        config.patience,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuner::features::FeatureTerm;

    #[test]
    fn aggregate_loss_and_gradient_match_expanded_labels() {
        let labels = LabelCounts {
            losses: 2,
            draws: 3,
            wins: 5,
        };
        let predicted = 0.63f64;
        let expanded = 2.0 * predicted.powi(2)
            + 3.0 * (predicted - 0.5).powi(2)
            + 5.0 * (predicted - 1.0).powi(2);
        assert!((sample_loss(predicted, labels) - expanded).abs() < 1.0e-12);
        let expanded_error =
            2.0 * (predicted - 0.0) + 3.0 * (predicted - 0.5) + 5.0 * (predicted - 1.0);
        let aggregate_error = labels.weight() as f64 * (predicted - labels.soft_target());
        assert!((expanded_error - aggregate_error).abs() < 1.0e-12);
    }

    #[test]
    fn probability_is_stable_at_extreme_scores() {
        assert_eq!(prediction(1.0e9, 1.0), 1.0);
        assert_eq!(prediction(-1.0e9, 1.0), 0.0);
        assert!((prediction(0.0, 1.0) - 0.5).abs() < f64::EPSILON);
    }

    fn small_partition() -> Partition {
        Partition {
            offsets: vec![0, 1, 2],
            entries: vec![
                FeatureTerm {
                    index: 0,
                    coefficient: 2.0,
                },
                FeatureTerm {
                    index: 1,
                    coefficient: -1.5,
                },
            ],
            labels: vec![
                LabelCounts {
                    losses: 1,
                    draws: 2,
                    wins: 4,
                },
                LabelCounts {
                    losses: 3,
                    draws: 1,
                    wins: 0,
                },
            ],
            total_weight: 11,
        }
    }

    #[test]
    fn sparse_gradient_matches_finite_difference() {
        let partition = small_partition();
        let mut parameters = [0.0; PARAMETER_COUNT];
        parameters[0] = 13.0;
        parameters[1] = -7.0;
        let analytic = accumulate_range(&partition, &parameters, 0.8, 0, 2, true);
        let epsilon = 1.0e-4;
        for index in [0usize, 1] {
            parameters[index] += epsilon;
            let plus = accumulate_range(&partition, &parameters, 0.8, 0, 2, false).loss;
            parameters[index] -= 2.0 * epsilon;
            let minus = accumulate_range(&partition, &parameters, 0.8, 0, 2, false).loss;
            parameters[index] += epsilon;
            let numeric = (plus - minus) / (2.0 * epsilon);
            assert!(
                (analytic.gradient[index] - numeric).abs() < 1.0e-8,
                "index {index}: analytic {}, numeric {numeric}",
                analytic.gradient[index]
            );
        }
    }

    #[test]
    fn fixed_chunks_make_thread_count_reproducible() {
        let partition = small_partition();
        let parameters = [0.25; PARAMETER_COUNT];
        let one = accumulate(&partition, &parameters, 0.8, 1, true);
        let many = accumulate(&partition, &parameters, 0.8, 4, true);
        assert_eq!(one.loss, many.loss);
        assert_eq!(one.gradient, many.gradient);
    }
}
