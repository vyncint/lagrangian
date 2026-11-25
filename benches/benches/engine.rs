//! The engine functions on their own: the double pendulum against its
//! hand-derived equations, and a chain of springs from 2 to 32 coordinates.

use std::hint::black_box;

use benches::{Counted, Springs, double_pendulum_by_hand, springs_state, time, wanted};
use lagrangian::ad::{Dual, Grad};
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;
use lagrangian::{linearize_free, mass_matrix};

fn double_pendulum() {
    let dp = DoublePendulum::default();
    let (q, dq) = ([1.1, -0.4], [0.7, 1.9]);
    let engine = accelerations(&dp, q, dq, 0.0).unwrap();
    let hand = double_pendulum_by_hand(&dp, q, dq);
    let agree = (engine[0] - hand[0]).abs().max((engine[1] - hand[1]).abs());

    let by_hand = time(|| double_pendulum_by_hand(&dp, black_box(q), black_box(dq)));
    let acc = time(|| accelerations(&dp, black_box(q), black_box(dq), 0.0));
    let terms = time(|| eom_terms(&dp, black_box(q), black_box(dq), 0.0));
    let mass = time(|| mass_matrix(&dp, black_box(q), black_box(dq), 0.0));
    let e = time(|| energy(&dp, black_box(q), black_box(dq), 0.0));
    let lin = time(|| linearize_free(&dp, black_box(q), black_box(dq), 0.0));
    println!("## engine: double pendulum (N = 2)");
    println!(
        "   accelerations      {acc:8.0} ns   ({:.1}x the hand-derived equations, {by_hand:.0} ns; they agree to {agree:.0e})",
        acc / by_hand
    );
    println!(
        "   eom_terms          {terms:8.0} ns   (mass_matrix {mass:.0} ns of it: {:.0}%)",
        100.0 * mass / terms
    );
    println!("   energy             {e:8.0} ns");
    println!("   linearize_free     {lin:8.0} ns");
}

fn springs<const N: usize>() {
    let sys = Springs::<N>;
    let (q, dq) = springs_state::<N>();
    let f64_pass = time(|| sys.lagrangian(black_box(q), black_box(dq), 0.0));
    let grad_pass = time(|| {
        sys.lagrangian(
            core::array::from_fn(|i| Grad::<N>::variable(black_box(q)[i], i)),
            dq.map(Grad::constant),
            Grad::constant(0.0),
        )
    });
    let dual_pass = time(|| {
        sys.lagrangian(
            black_box(q).map(|x| Dual::constant(Grad::<N>::constant(x))),
            core::array::from_fn(|i| Dual::new(Grad::variable(dq[i], i), Grad::constant(1.0))),
            Dual::constant(Grad::constant(0.0)),
        )
    });
    let acc = time(|| accelerations(&sys, black_box(q), black_box(dq), 0.0));
    let mass = time(|| mass_matrix(&sys, black_box(q), black_box(dq), 0.0));
    let lin = if N <= 16 {
        format!(
            "{:9.1}",
            time(|| linearize_free(&sys, black_box(q), black_box(dq), 0.0)) / 1e3
        )
    } else {
        format!("{:>9}", "-")
    };
    let counted = Counted::new(sys);
    accelerations(&counted, q, dq, 0.0).unwrap();
    let per_call = counted.take();
    println!(
        "   {N:>3} | {:>9.3} | {:>9.3} | {:>9.3} | {:>9.2} | {:>5.0}% | {lin} | {}",
        f64_pass / 1e3,
        grad_pass / 1e3,
        dual_pass / 1e3,
        acc / 1e3,
        100.0 * mass / acc,
        per_call.per(1.0)
    );
}

fn main() {
    if wanted("engine/double_pendulum") {
        double_pendulum();
    }
    if wanted("engine/springs") {
        println!("## engine: chain of N springs, one cosine per coordinate (µs)");
        println!(
            "     N | L at f64  | Grad<N>   | Dual<Grad>| accel.    | in M   | linearize | evaluations per accelerations call"
        );
        springs::<2>();
        springs::<4>();
        springs::<8>();
        springs::<16>();
        springs::<32>();
    }
}
