use std::{env, process::ExitCode};

use bullet_lib::{
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::{LocalSettings, TestDataset},
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use lattice_nnue_trainer::{FEATURES, HIDDEN, HalfKp, QA, QB, SCALE};

fn usage() -> ExitCode {
    eprintln!(
        "usage: lattice-nnue-trainer TRAIN_DATA[,TRAIN_DATA...] OUTPUT_DIR [smoke|full] [VALIDATION_DATA]"
    );
    ExitCode::FAILURE
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(data) = args.get(1) else {
        return usage();
    };
    let Some(output) = args.get(2) else {
        return usage();
    };
    let mode = args.get(3).map_or("smoke", String::as_str);
    let (batches_per_superbatch, end_superbatch, save_rate) = match mode {
        "smoke" => (64, 4, 1),
        "full" => (3_052, 20, 2),
        _ => return usage(),
    };

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(HalfKp)
        .save_format(&[
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", FEATURES, HIDDEN);
            let l1 = builder.new_affine("l1", 2 * HIDDEN, 1);
            let stm = l0.forward(stm_inputs).screlu();
            let ntm = l0.forward(ntm_inputs).screlu();
            l1.forward(stm.concat(ntm))
        });

    let schedule = TrainingSchedule {
        net_id: format!("lattice-halfkp-256-{mode}"),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch,
            start_superbatch: 1,
            end_superbatch,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: if mode == "smoke" { 0.0003 } else { 0.00001 },
            final_superbatch: end_superbatch,
        },
        save_rate,
    };

    let test_set = args.get(4).map(|path| TestDataset {
        path,
        freq: if mode == "smoke" { 8 } else { 64 },
    });
    let settings = LocalSettings {
        threads: 4,
        test_set,
        output_directory: output,
        batch_queue_size: 32,
    };
    let training_paths: Vec<&str> = data.split(',').filter(|path| !path.is_empty()).collect();
    if training_paths.is_empty() {
        return usage();
    }
    let loader = DirectSequentialDataLoader::new(&training_paths);
    trainer.run(&schedule, &settings, &loader);
    ExitCode::SUCCESS
}
