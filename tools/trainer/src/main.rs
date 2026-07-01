//! Train Lattice's NNUE with bullet.
//!
//! Architecture: `(768 -> 256) x2 -> 1`, dual perspective, SCReLU, quantised
//! `i16` (QA = 255, QB = 64), eval scale 400. These MUST match the engine's
//! `board::nnue` loader; change them in both places together.
//!
//! Usage: `cargo run --release --features metal -- <data.data> [superbatches] [batches_per_superbatch]`
//!
//! The quantised network is written to `checkpoints/lattice-<sb>/quantised.bin`;
//! copy the final one to `src/board/net.nnue` in the engine.

use bullet::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const HIDDEN_SIZE: usize = 256;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "data.data".to_string());
    // Small defaults so a smoke run finishes in seconds; scale both up (e.g.
    // 6104 batches x 40 superbatches) for a real net over tens of millions.
    let superbatches: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let batches_per_superbatch: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    let schedule = TrainingSchedule {
        net_id: "lattice".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.4 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.3,
            step: superbatches / 3 + 1,
        },
        save_rate: superbatches,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[&data_path]);

    trainer.run(&schedule, &settings, &data_loader);
}
