use std::{env, fs, io::Read, process::ExitCode};

use bulletformat::ChessBoard;
use lattice_nnue_trainer::{QuantizedNetwork, SCALE};

fn sigmoid(score: f64) -> f64 {
    1.0 / (1.0 + (-score / f64::from(SCALE)).exp())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (Some(network_path), Some(data_path)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: eval-network LATTICE_NNUE BULLET_DATA [LIMIT]");
        return ExitCode::FAILURE;
    };
    let limit = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(100_000);
    let network = match fs::read(network_path)
        .map_err(|error| error.to_string())
        .and_then(|bytes| QuantizedNetwork::parse(&bytes))
    {
        Ok(network) => network,
        Err(error) => {
            eprintln!("failed to load network: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut data = match fs::File::open(data_path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!("failed to open data: {error}");
            return ExitCode::FAILURE;
        }
    };
    let mut count = 0_usize;
    let mut loss = 0.0_f64;
    let mut absolute_cp_error = 0.0_f64;
    let mut record = [0_u8; 32];
    while count < limit {
        match data.read_exact(&mut record) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => {
                eprintln!("failed at record {count}: {error}");
                return ExitCode::FAILURE;
            }
        }
        // ChessBoard is repr(C), exactly 32 bytes, and every bit pattern in
        // these integer fields is valid. The shard was produced by the pinned
        // bulletformat writer and is separately content-addressed by Nyquist.
        let board: ChessBoard = unsafe { std::mem::transmute(record) };
        let eval = network.evaluate(&board);
        let target =
            0.75 * (f64::from(board.result) / 2.0) + 0.25 * sigmoid(f64::from(board.score));
        let prediction = sigmoid(f64::from(eval));
        loss += (prediction - target).powi(2);
        absolute_cp_error += f64::from((eval - i32::from(board.score)).abs());
        count += 1;
    }
    if count == 0 {
        eprintln!("dataset contained no complete records");
        return ExitCode::FAILURE;
    }
    println!("records={count}");
    println!("blended_mse={:.8}", loss / count as f64);
    println!("score_mae_cp={:.3}", absolute_cp_error / count as f64);
    ExitCode::SUCCESS
}
