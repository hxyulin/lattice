use std::{env, fs, process::ExitCode};

use lattice_nnue_trainer::{NetworkArchitecture, pack_network_for};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let (Some(input), Some(output)) = (args.get(1), args.get(2)) else {
        eprintln!("usage: pack-network BULLET_QUANTISED_BIN LATTICE_NNUE [halfkp|chess768]");
        return ExitCode::FAILURE;
    };
    let architecture = match args.get(3).map(String::as_str).unwrap_or("halfkp") {
        "halfkp" => NetworkArchitecture::HalfKp,
        "chess768" => NetworkArchitecture::Chess768,
        other => {
            eprintln!("unsupported architecture: {other}");
            return ExitCode::FAILURE;
        }
    };
    let raw = match fs::read(input) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to read {input}: {error}");
            return ExitCode::FAILURE;
        }
    };
    let packed = match pack_network_for(&raw, architecture) {
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
