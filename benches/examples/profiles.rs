//! What a crate that depends on `lagrangian` gets from each build profile.
//!
//! This member is such a crate, so building this example the ways below
//! reproduces the table in the README and in `PLAN.md` §7.3:
//!
//! ```sh
//! cargo run -p benches --example profiles --release
//! cargo run -p benches --example profiles
//! cargo run -p benches --example profiles --config 'profile.dev.package.lagrangian.opt-level=3'
//! cargo run -p benches --example profiles --config 'profile.dev.opt-level=1'
//! cargo run -p benches --example profiles --config 'profile.dev.opt-level=2'
//! ```

use std::hint::black_box;
use std::time::Instant;

use benches::time;
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

fn main() {
    let pendulum = DoublePendulum::default();
    let (q, dq) = ([1.1, -0.4], [0.7, 1.9]);
    let per_call = time(|| accelerations(&pendulum, black_box(q), black_box(dq), 0.0));
    let mut runs: Vec<f64> = (0..7)
        .map(|_| {
            let start = Instant::now();
            let config = SimConfig::default();
            black_box(simulate(&pendulum, (0.0, 20.0), [2.2, 2.8], [0.0, 0.0], &config).unwrap());
            start.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    runs.sort_by(f64::total_cmp);
    println!(
        "accelerations, double pendulum: {per_call:.0} ns; simulate, t in [0, 20]: {:.1} ms",
        runs[3]
    );
}
