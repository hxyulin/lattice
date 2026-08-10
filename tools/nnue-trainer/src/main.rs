use std::{env, process::ExitCode};

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::{LocalSettings, TestDataset},
    },
    value::{ValueTrainerBuilder, loader::DirectSequentialDataLoader},
};
use lattice_nnue_trainer::{FEATURES, HIDDEN, HalfKp, QA, QB, SCALE};

macro_rules! build_trainer {
    ($inputs:expr, $input_count:expr) => {{
        ValueTrainerBuilder::default()
            .dual_perspective()
            .optimiser(AdamW)
            .inputs($inputs)
            .save_format(&[
                SavedFormat::id("l0b").round().quantise::<i16>(QA),
                SavedFormat::id("l0w").round().quantise::<i16>(QA),
                SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
                SavedFormat::id("l1w").round().quantise::<i16>(QB),
            ])
            .loss_fn(|output, target| output.sigmoid().squared_error(target))
            .build(|builder, stm_inputs, ntm_inputs| {
                let l0 = builder.new_affine("l0", $input_count, HIDDEN);
                let l1 = builder.new_affine("l1", 2 * HIDDEN, 1);
                let stm = l0.forward(stm_inputs).screlu();
                let ntm = l0.forward(ntm_inputs).screlu();
                l1.forward(stm.concat(ntm))
            })
    }};
}

fn usage() -> ExitCode {
    eprintln!(
        "usage: lattice-nnue-trainer TRAIN_DATA[,TRAIN_DATA...] OUTPUT_DIR \
         [smoke|full|controlled-halfkp|controlled-chess768] [VALIDATION_DATA]"
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
        "controlled-halfkp" | "controlled-chess768" => (3_052, 6, 1),
        _ => return usage(),
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
    let steps = TrainingSteps {
        batch_size: 16_384,
        batches_per_superbatch,
        start_superbatch: 1,
        end_superbatch,
    };
    let controlled = mode.starts_with("controlled-");
    if mode == "controlled-chess768" {
        let mut trainer = build_trainer!(Chess768, 768);
        let schedule = TrainingSchedule {
            net_id: "lattice-chess768-256-controlled".to_string(),
            eval_scale: SCALE as f32,
            steps,
            wdl_scheduler: wdl::ConstantWDL { value: 0.40 },
            lr_scheduler: lr::StepLR {
                start: 0.001,
                gamma: 0.3,
                step: end_superbatch / 3 + 1,
            },
            save_rate,
        };
        trainer.run(
            &schedule,
            &settings,
            &DirectSequentialDataLoader::new(&training_paths),
        );
    } else {
        let mut trainer = build_trainer!(HalfKp, FEATURES);
        if controlled {
            let schedule = TrainingSchedule {
                net_id: "lattice-halfkp-256-controlled".to_string(),
                eval_scale: SCALE as f32,
                steps,
                wdl_scheduler: wdl::ConstantWDL { value: 0.40 },
                lr_scheduler: lr::StepLR {
                    start: 0.001,
                    gamma: 0.3,
                    step: end_superbatch / 3 + 1,
                },
                save_rate,
            };
            trainer.run(
                &schedule,
                &settings,
                &DirectSequentialDataLoader::new(&training_paths),
            );
        } else {
            let schedule = TrainingSchedule {
                net_id: format!("lattice-halfkp-256-{mode}"),
                eval_scale: SCALE as f32,
                steps,
                wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
                lr_scheduler: lr::CosineDecayLR {
                    initial_lr: 0.001,
                    final_lr: if mode == "smoke" { 0.0003 } else { 0.00001 },
                    final_superbatch: end_superbatch,
                },
                save_rate,
            };
            trainer.run(
                &schedule,
                &settings,
                &DirectSequentialDataLoader::new(&training_paths),
            );
        }
    }
    ExitCode::SUCCESS
}
