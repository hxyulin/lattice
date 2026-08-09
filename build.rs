//! Selects a trained NNUE or generates a deterministic integration fixture.

use std::{env, fs, io::Write, path::PathBuf};

const MAGIC: &[u8; 8] = b"LTNNUE01";
const VERSION: u32 = 1;
const FEATURE_ABI: u32 = 1;
const INPUTS: usize = 40_960;
const HIDDEN: usize = 256;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn push_i16(buf: &mut Vec<u8>, value: i16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn fixture_weight(feature: usize, neuron: usize) -> i16 {
    let mixed = (feature as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .rotate_left((neuron & 63) as u32)
        ^ (neuron as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    ((mixed % 7) as i16) - 3
}

fn write_bootstrap(path: &PathBuf) {
    let payload_len = 2 * (HIDDEN + INPUTS * HIDDEN + 1 + 2 * HIDDEN);
    let mut payload = Vec::with_capacity(payload_len);

    for neuron in 0..HIDDEN {
        push_i16(&mut payload, (neuron as i16 % 5) - 2);
    }
    for feature in 0..INPUTS {
        for neuron in 0..HIDDEN {
            push_i16(&mut payload, fixture_weight(feature, neuron));
        }
    }
    push_i16(&mut payload, 0);
    for neuron in 0..2 * HIDDEN {
        push_i16(&mut payload, (neuron as i16 % 9) - 4);
    }
    assert_eq!(payload.len(), payload_len);

    let mut file = fs::File::create(path).expect("create bootstrap NNUE");
    file.write_all(MAGIC).unwrap();
    for value in [VERSION, FEATURE_ABI, INPUTS as u32, HIDDEN as u32] {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    for value in [QA, QB, SCALE, 0] {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    file.write_all(&(payload_len as u64).to_le_bytes()).unwrap();
    file.write_all(&fnv1a(&payload).to_le_bytes()).unwrap();
    file.write_all(&payload).unwrap();
}

fn main() {
    println!("cargo:rerun-if-changed=networks/lattice.nnue");
    println!("cargo:rerun-if-env-changed=LATTICE_NNUE_PATH");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let trained = manifest.join("networks/lattice.nnue");
    let selected = if let Some(path) = env::var_os("LATTICE_NNUE_PATH") {
        let path = PathBuf::from(path);
        assert!(path.is_file(), "LATTICE_NNUE_PATH must name a network file");
        println!("cargo:rerun-if-changed={}", path.display());
        path
    } else if trained.is_file() {
        trained
    } else {
        let generated = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("bootstrap.nnue");
        write_bootstrap(&generated);
        println!(
            "cargo:warning=using deterministic bootstrap NNUE; production evaluation remains HCE"
        );
        generated
    };
    println!("cargo:rustc-env=LATTICE_NNUE_FILE={}", selected.display());
}
