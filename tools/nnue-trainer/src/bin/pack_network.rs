use std::{env, fs, process::ExitCode};

use lattice_nnue_trainer::pack_network;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: pack-network BULLET_QUANTISED_BIN LATTICE_NNUE");
        return ExitCode::FAILURE;
    };
    let raw = match fs::read(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {input}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let packed = match pack_network(&raw) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to pack network: {error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(output, packed) {
        eprintln!("failed to write {output}: {error}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}
