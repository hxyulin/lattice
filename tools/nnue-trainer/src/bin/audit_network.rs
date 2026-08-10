use std::{env, fs, io::Read, process::ExitCode};

use bullet_lib::game::inputs::SparseInputType;
use bulletformat::ChessBoard;
use lattice::{Board, Color};
use lattice_nnue_trainer::{FloatNetwork, HalfKp, QuantizedNetwork, SCALE, pack_network};

fn usage() -> ExitCode {
    eprintln!("usage: audit-network RAW_BIN QUANTISED_BIN LATTICE_NNUE BULLET_DATA [LIMIT]");
    ExitCode::FAILURE
}

fn sigmoid(score: f64) -> f64 {
    1.0 / (1.0 + (-score / f64::from(SCALE)).exp())
}

fn bullet_features(board: &ChessBoard) -> (Vec<usize>, Vec<usize>) {
    let (mut stm, mut ntm) = (Vec::new(), Vec::new());
    HalfKp.map_features(board, |a, b| {
        stm.push(a);
        ntm.push(b);
    });
    stm.sort_unstable();
    ntm.sort_unstable();
    (stm, ntm)
}

fn engine_features(board: &Board) -> (Vec<usize>, Vec<usize>) {
    let stm = board.state().side_to_move();
    let for_side = |perspective: Color| {
        let king = board.king_square(perspective);
        let mut result = Vec::new();
        for square in board.occupied() {
            let piece = board.piece_on(square).unwrap();
            if let Some(index) = lattice::nnue::feature_index(perspective, king, piece, square) {
                result.push(index);
            }
        }
        result.sort_unstable();
        result
    };
    (for_side(stm), for_side(stm.flip()))
}

fn audit_fixtures(network: &QuantizedNetwork) -> Result<(), String> {
    const FENS: &[&str] = &[
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
        "r3k2r/pp1n1ppp/2pbpn2/q7/2BPP3/2N2N2/PPQ2PPP/R3K2R b KQkq - 3 12",
        "8/2P2k2/3p4/4P3/8/5K2/8/8 w - - 0 45",
        "4k3/1Q6/8/8/8/8/6q1/4K3 b - - 0 1",
    ];
    let mut fixture_scores = Vec::new();
    for fen in FENS {
        let engine: Board = fen
            .parse()
            .map_err(|error| format!("bad fixture: {error}"))?;
        let bullet: ChessBoard = format!("{fen} | 0 | 0.5").parse()?;
        let expected = engine_features(&engine);
        let actual = bullet_features(&bullet);
        if actual != expected {
            return Err(format!("feature ABI mismatch for {fen}"));
        }
        let packed = network.evaluate(&bullet);
        let runtime = lattice::eval::evaluate(&engine);
        if packed != runtime {
            return Err(format!(
                "runtime mismatch for {fen}: independent={packed}, engine={runtime}"
            ));
        }
        fixture_scores.push(runtime);
    }
    println!("fixture_feature_abi=PASS ({})", FENS.len());
    println!("fixture_engine_inference=PASS ({})", FENS.len());
    println!("fixture_scores_cp={fixture_scores:?}");
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let (Some(raw_path), Some(quantised_path), Some(packed_path), Some(data_path)) =
        (args.get(1), args.get(2), args.get(3), args.get(4))
    else {
        return Err("missing arguments".to_string());
    };
    let limit = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(20_000);

    let raw = fs::read(raw_path).map_err(|error| error.to_string())?;
    let quantised = fs::read(quantised_path).map_err(|error| error.to_string())?;
    let packed = fs::read(packed_path).map_err(|error| error.to_string())?;
    let float = FloatNetwork::parse(&raw)?;
    let expected_packed = pack_network(&quantised)?;
    if expected_packed != packed {
        return Err("packed network is not the supplied Bullet checkpoint".to_string());
    }
    println!("checkpoint_payload=PASS");
    let integer = QuantizedNetwork::parse(&packed)?;
    audit_fixtures(&integer)?;

    let mut data = fs::File::open(data_path).map_err(|error| error.to_string())?;
    let mut record = [0_u8; 32];
    let mut count = 0_usize;
    let mut float_loss = 0.0_f64;
    let mut integer_loss = 0.0_f64;
    let mut error_sum = 0.0_f64;
    let mut squared_error_sum = 0.0_f64;
    let mut max_error = 0.0_f64;
    let mut sign_matches = 0_usize;
    let mut accumulator_min = i32::MAX;
    let mut accumulator_max = i32::MIN;
    let mut label_abs_sum = 0.0_f64;
    let mut prediction_abs_sum = 0.0_f64;
    let mut mate_like = 0_usize;
    let mut decisive_score = 0_usize;
    let mut score_sign_matches = 0_usize;
    let mut score_x = 0.0_f64;
    let mut score_y = 0.0_f64;
    let mut score_xx = 0.0_f64;
    let mut score_yy = 0.0_f64;
    let mut score_xy = 0.0_f64;
    while count < limit {
        match data.read_exact(&mut record) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(format!("dataset record {count}: {error}")),
        }
        // The dataset and dependency are pinned to bulletformat 1.8, whose
        // repr(C) ChessBoard is exactly 32 bytes of integer fields.
        let board: ChessBoard = unsafe { std::mem::transmute(record) };
        let float_cp = f64::from(float.evaluate_cp(&board));
        let integer_cp = f64::from(integer.evaluate(&board));
        let target =
            0.75 * (f64::from(board.result) / 2.0) + 0.25 * sigmoid(f64::from(board.score));
        float_loss += (sigmoid(float_cp) - target).powi(2);
        integer_loss += (sigmoid(integer_cp) - target).powi(2);
        let error = (float_cp - integer_cp).abs();
        error_sum += error;
        squared_error_sum += error * error;
        max_error = max_error.max(error);
        sign_matches += usize::from(float_cp.signum() == integer_cp.signum());
        let (lo, hi) = integer.accumulator_bounds(&board);
        accumulator_min = accumulator_min.min(lo);
        accumulator_max = accumulator_max.max(hi);
        let label = f64::from(board.score);
        label_abs_sum += label.abs();
        prediction_abs_sum += float_cp.abs();
        mate_like += usize::from(label.abs() >= 30_000.0);
        if label != 0.0 {
            decisive_score += 1;
            score_sign_matches += usize::from(label.signum() == float_cp.signum());
        }
        score_x += label;
        score_y += float_cp;
        score_xx += label * label;
        score_yy += float_cp * float_cp;
        score_xy += label * float_cp;
        count += 1;
    }
    if count == 0 {
        return Err("dataset contained no complete records".to_string());
    }
    let n = count as f64;
    println!("records={count}");
    println!("float_blended_mse={:.8}", float_loss / n);
    println!("integer_blended_mse={:.8}", integer_loss / n);
    println!("quantisation_mae_cp={:.4}", error_sum / n);
    println!("quantisation_rmse_cp={:.4}", (squared_error_sum / n).sqrt());
    println!("quantisation_max_cp={max_error:.4}");
    println!(
        "quantisation_sign_agreement={:.4}%",
        100.0 * sign_matches as f64 / n
    );
    println!("accumulator_range=[{accumulator_min},{accumulator_max}]");
    println!(
        "accumulator_i16_safe={}",
        accumulator_min >= i32::from(i16::MIN) && accumulator_max <= i32::from(i16::MAX)
    );
    let covariance = score_xy - score_x * score_y / n;
    let variance_x = score_xx - score_x * score_x / n;
    let variance_y = score_yy - score_y * score_y / n;
    let correlation = covariance / (variance_x * variance_y).sqrt();
    println!("label_mean_abs_cp={:.3}", label_abs_sum / n);
    println!("prediction_mean_abs_cp={:.3}", prediction_abs_sum / n);
    println!("label_mate_like={:.4}%", 100.0 * mate_like as f64 / n);
    println!("prediction_label_correlation={correlation:.6}");
    println!(
        "prediction_label_sign_agreement={:.4}%",
        100.0 * score_sign_matches as f64 / decisive_score as f64
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("audit failed: {error}");
            usage()
        }
    }
}
