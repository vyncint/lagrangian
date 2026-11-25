//! The solvers on the chaotic double pendulum: time and Lagrangian
//! evaluations per step, and the energy error they buy.

use std::hint::black_box;
use std::time::Instant;

use benches::{Counted, time, wanted};
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

const START: ([f64; 2], [f64; 2]) = ([2.2, 2.8], [0.0, 0.0]);

fn configs() -> [(&'static str, SimConfig); 5] {
    [
        ("Dopri5, default", SimConfig::default()),
        (
            "Dopri5, tight",
            SimConfig::dopri5().with_tolerances(1e-12, 1e-14),
        ),
        ("Variational2, dt 0.005", SimConfig::variational2(0.005)),
        ("Variational4, dt 0.01", SimConfig::variational4(0.01)),
        ("Variational4, dt 0.005", SimConfig::variational4(0.005)),
    ]
}

fn chaotic() {
    let dp = DoublePendulum::default();
    println!("## steppers: chaotic double pendulum, t in [0, 20]");
    println!(
        "   {:<24} | {:>6} | {:>8} | {:>7} | {:>12} | evaluations per step",
        "solver", "steps", "ms", "µs/step", "energy error"
    );
    for (label, config) in configs() {
        let mut runs: Vec<f64> = (0..7)
            .map(|_| {
                let start = Instant::now();
                black_box(simulate(&dp, (0.0, 20.0), START.0, START.1, &config).unwrap());
                start.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        runs.sort_by(f64::total_cmp);
        let ms = runs[3];
        let counted = Counted::new(dp);
        let traj = simulate(&counted, (0.0, 20.0), START.0, START.1, &config).unwrap();
        let steps = traj.len_segments() as f64;
        let drift = traj.energy_drift_at_knots(&dp).max_rel_drift;
        println!(
            "   {label:<24} | {steps:>6} | {ms:>8.2} | {:>7.2} | {drift:>12.1e} | {}",
            ms * 1e3 / steps,
            counted.take().per(steps)
        );
    }
}

fn one_step() {
    let dp = DoublePendulum::default();
    println!("## steppers: one step from the same moving state");
    for (label, config) in configs() {
        let config = if label.starts_with("Dopri5") {
            config.with_initial_step(0.005)
        } else {
            config
        };
        let counted = Counted::new(dp);
        let mut stepper = Stepper::new(&counted, 0.0, [1.2, 2.0], [0.4, -0.9], config).unwrap();
        counted.take();
        stepper.step().unwrap();
        let counts = counted.take();
        let ns = time(|| {
            let mut s = Stepper::new(dp, 0.0, [1.2, 2.0], [0.4, -0.9], config).unwrap();
            s.step().unwrap()
        });
        println!(
            "   {label:<24} | {:>3} evaluations ({}) | {:.2} µs with the stepper's construction",
            counts.total(),
            counts.per(1.0),
            ns / 1e3
        );
    }
}

fn main() {
    if wanted("steppers/chaotic") {
        chaotic();
    }
    if wanted("steppers/one_step") {
        one_step();
    }
}
