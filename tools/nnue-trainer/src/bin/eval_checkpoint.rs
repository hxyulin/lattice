use std::{env, fs, io::Read, process::ExitCode};

use bullet_lib::game::inputs::{Chess768, SparseInputType};
use bulletformat::ChessBoard;
use lattice_nnue_trainer::{HIDDEN, HalfKp, QA, QB, SCALE};

#[derive(Clone, Copy)]
enum Architecture {
    Chess768,
    HalfKp,
}

impl Architecture {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "chess768" => Some(Self::Chess768),
            "halfkp" => Some(Self::HalfKp),
            _ => None,
        }
    }

    fn inputs(self) -> usize {
        match self {
            Self::Chess768 => 768,
            Self::HalfKp => 40_960,
        }
    }

    fn map_features(self, board: &ChessBoard, f: impl FnMut(usize, usize)) {
        match self {
            Self::Chess768 => Chess768.map_features(board, f),
            Self::HalfKp => HalfKp.map_features(board, f),
        }
    }
}

struct Network<T> {
    architecture: Architecture,
    feature_bias: Vec<T>,
    feature_weights: Vec<T>,
    output_bias: T,
    output_weights: Vec<T>,
}

impl Network<f32> {
    fn parse(architecture: Architecture, bytes: &[u8]) -> Result<Self, String> {
        let values = HIDDEN + architecture.inputs() * HIDDEN + 1 + 2 * HIDDEN;
        if bytes.len() != 4 * values {
            return Err(format!("raw size {}, expected {}", bytes.len(), 4 * values));
        }
        let mut cursor = 0;
        let mut next = || {
            let value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            value
        };
        Ok(Self {
            architecture,
            feature_bias: (0..HIDDEN).map(|_| next()).collect(),
            feature_weights: (0..architecture.inputs() * HIDDEN)
                .map(|_| next())
                .collect(),
            output_bias: next(),
            output_weights: (0..2 * HIDDEN).map(|_| next()).collect(),
        })
    }

    fn evaluate(&self, board: &ChessBoard) -> f64 {
        let mut stm: [f32; HIDDEN] = self.feature_bias.as_slice().try_into().unwrap();
        let mut ntm = stm;
        self.architecture.map_features(board, |a, b| {
            for neuron in 0..HIDDEN {
                stm[neuron] += self.feature_weights[a * HIDDEN + neuron];
                ntm[neuron] += self.feature_weights[b * HIDDEN + neuron];
            }
        });
        let activate = |x: f32| x.clamp(0.0, 1.0).powi(2);
        let mut output = self.output_bias;
        for (&x, &weight) in stm.iter().zip(&self.output_weights[..HIDDEN]) {
            output += activate(x) * weight;
        }
        for (&x, &weight) in ntm.iter().zip(&self.output_weights[HIDDEN..]) {
            output += activate(x) * weight;
        }
        f64::from(output) * f64::from(SCALE)
    }
}

impl Network<i16> {
    fn parse(architecture: Architecture, bytes: &[u8]) -> Result<Self, String> {
        let values = HIDDEN + architecture.inputs() * HIDDEN + 1 + 2 * HIDDEN;
        let required = 2 * values;
        if bytes.len() < required || bytes.len() - required >= 64 {
            return Err(format!("quantised size {} is invalid", bytes.len()));
        }
        let padding = &bytes[required..];
        if padding
            .iter()
            .enumerate()
            .any(|(index, &byte)| byte != b"bullet"[index % 6])
        {
            return Err("invalid Bullet padding".to_string());
        }
        let mut cursor = 0;
        let mut next = || {
            let value = i16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap());
            cursor += 2;
            value
        };
        Ok(Self {
            architecture,
            feature_bias: (0..HIDDEN).map(|_| next()).collect(),
            feature_weights: (0..architecture.inputs() * HIDDEN)
                .map(|_| next())
                .collect(),
            output_bias: next(),
            output_weights: (0..2 * HIDDEN).map(|_| next()).collect(),
        })
    }

    fn evaluate(&self, board: &ChessBoard) -> i32 {
        let bias: [i16; HIDDEN] = self.feature_bias.as_slice().try_into().unwrap();
        let mut stm = bias.map(i32::from);
        let mut ntm = stm;
        self.architecture.map_features(board, |a, b| {
            for neuron in 0..HIDDEN {
                stm[neuron] += i32::from(self.feature_weights[a * HIDDEN + neuron]);
                ntm[neuron] += i32::from(self.feature_weights[b * HIDDEN + neuron]);
            }
        });
        let mut output = 0_i64;
        for (&x, &weight) in stm.iter().zip(&self.output_weights[..HIDDEN]) {
            let x = i64::from(x).clamp(0, i64::from(QA));
            output += x * x * i64::from(weight);
        }
        for (&x, &weight) in ntm.iter().zip(&self.output_weights[HIDDEN..]) {
            let x = i64::from(x).clamp(0, i64::from(QA));
            output += x * x * i64::from(weight);
        }
        output /= i64::from(QA);
        output += i64::from(self.output_bias);
        output *= i64::from(SCALE);
        output /= i64::from(QA) * i64::from(QB);
        output as i32
    }
}

fn sigmoid(score: f64) -> f64 {
    1.0 / (1.0 + (-score / f64::from(SCALE)).exp())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    let (Some(architecture), Some(raw), Some(quantised), Some(data)) =
        (args.get(1), args.get(2), args.get(3), args.get(4))
    else {
        return Err("missing arguments".to_string());
    };
    let architecture = Architecture::parse(architecture)
        .ok_or_else(|| "architecture must be chess768 or halfkp".to_string())?;
    let limit = args
        .get(5)
        .map(|x| x.parse::<usize>())
        .transpose()
        .map_err(|error| error.to_string())?
        .unwrap_or(100_000);
    let float = Network::<f32>::parse(
        architecture,
        &fs::read(raw).map_err(|error| error.to_string())?,
    )?;
    let integer = Network::<i16>::parse(
        architecture,
        &fs::read(quantised).map_err(|error| error.to_string())?,
    )?;
    let mut reader = fs::File::open(data).map_err(|error| error.to_string())?;
    let mut record = [0_u8; 32];
    let mut count = 0_usize;
    let mut float_loss = 0.0;
    let mut integer_loss = 0.0;
    let mut quantisation_error = 0.0;
    while count < limit {
        match reader.read_exact(&mut record) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(error) => return Err(error.to_string()),
        }
        // Both producer and consumer pin bulletformat 1.8's 32-byte repr(C).
        let board: ChessBoard = unsafe { std::mem::transmute(record) };
        let float_cp = float.evaluate(&board);
        let integer_cp = integer.evaluate(&board);
        let target =
            0.40 * (f64::from(board.result) / 2.0) + 0.60 * sigmoid(f64::from(board.score));
        float_loss += (sigmoid(float_cp) - target).powi(2);
        integer_loss += (sigmoid(f64::from(integer_cp)) - target).powi(2);
        quantisation_error += (float_cp - f64::from(integer_cp)).abs();
        count += 1;
    }
    if count == 0 {
        return Err("dataset contained no records".to_string());
    }
    let n = count as f64;
    println!(
        "records\tfloat_mse\tinteger_mse\tquantisation_mae_cp\n{count}\t{:.8}\t{:.8}\t{:.3}",
        float_loss / n,
        integer_loss / n,
        quantisation_error / n
    );
    Ok(())
}

fn main() -> ExitCode {
    if let Err(error) = run() {
        eprintln!("{error}\nusage: eval-checkpoint ARCH RAW_BIN QUANTISED_BIN BULLET_DATA [LIMIT]");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
