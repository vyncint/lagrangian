//! The stack each operation needs, held to the table in the crate
//! documentation (`lib.rs`, "Stack") and the README.
//!
//! Each operation runs in a thread with half again the stack the table
//! gives for it, in the profile this test was built with: the release
//! column under `cargo test --release`, the debug one otherwise. An overflow
//! aborts the whole process, so each case runs in a child, which is this
//! test binary run again with the case in an environment variable.

use std::hint::black_box;
use std::process::Command;

use lagrangian::linearize_free;
use lagrangian::prelude::*;

/// `N` masses in a line tied by springs, the system the table was measured
/// on: one cosine per coordinate, and a mass matrix that depends on the
/// configuration (`benches/src/lib.rs` has the same).
struct Springs<const N: usize>;

impl<const N: usize> Lagrangian<N> for Springs<N> {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], _t: S) -> S {
        let mut total = S::ZERO;
        let mut left = S::ZERO;
        for i in 0..N {
            let mass = q[i].cos() * 0.25 + 1.0;
            let stretch = q[i] - left;
            let s2 = stretch * stretch;
            total += dq[i] * dq[i] * mass - s2 - s2 * s2 * 0.1;
            left = q[i];
        }
        total * 0.5
    }
}

const OPERATIONS: [&str; 5] = [
    "accelerations",
    "eom_terms",
    "Dopri5 step",
    "Variational4 step",
    "linearize_free",
];

/// The table, in KiB, for the operations above in that order. Linux gives
/// a thread no less than 16 KiB, so a need below that reads 16.
const RELEASE: [(usize, [usize; 5]); 3] = [
    (8, [16, 16, 16, 16, 16]),
    (16, [38, 32, 47, 64, 83]),
    (32, [189, 166, 206, 279, 470]),
];
const DEBUG: [(usize, [usize; 5]); 3] = [
    (8, [33, 32, 41, 45, 64]),
    (16, [87, 83, 99, 112, 197]),
    (32, [292, 279, 304, 362, 693]),
];

/// Names the case a child is to run.
const CASE: &str = "LAGRANGIAN_STACK_CASE";

// Its own frame per N: inlined into `dispatch`, every size would share the
// frame of the largest.
#[inline(never)]
fn run<const N: usize>(operation: usize) {
    let sys = Springs::<N>;
    let q: [f64; N] = core::array::from_fn(|i| 0.3 + 0.01 * i as f64);
    let dq: [f64; N] = core::array::from_fn(|i| 0.2 - 0.01 * i as f64);
    match operation {
        0 => {
            accelerations(&sys, q, dq, 0.0).unwrap();
        }
        1 => {
            black_box(eom_terms(&sys, q, dq, 0.0));
        }
        2 => {
            let mut stepper = Stepper::new(&sys, 0.0, q, dq, SimConfig::default()).unwrap();
            stepper.step().unwrap();
        }
        3 => {
            let mut stepper =
                Stepper::new(&sys, 0.0, q, dq, SimConfig::variational4(0.01)).unwrap();
            stepper.step().unwrap();
        }
        _ => {
            linearize_free(&sys, q, dq, 0.0).unwrap();
        }
    }
}

fn dispatch(operation: usize, n: usize) {
    match n {
        8 => run::<8>(operation),
        16 => run::<16>(operation),
        32 => run::<32>(operation),
        _ => panic!("N = {n} is not in the table"),
    }
}

#[test]
fn every_operation_fits_in_the_documented_stack() {
    let (table, profile) = if cfg!(debug_assertions) {
        (DEBUG, "debug")
    } else {
        (RELEASE, "release")
    };
    let me = std::env::current_exe().unwrap();
    let mut failures = Vec::new();
    for (n, needs) in table {
        for (operation, kib) in needs.into_iter().enumerate() {
            let bytes = kib * 1024 * 3 / 2;
            let output = Command::new(&me)
                .args(["--exact", "child", "--test-threads=1"])
                .env(CASE, format!("{operation} {n} {bytes}"))
                .output()
                .unwrap();
            if !output.status.success() {
                failures.push(format!(
                    "{} at N = {n} did not fit in {} KiB ({profile}): {}",
                    OPERATIONS[operation],
                    bytes / 1024,
                    String::from_utf8_lossy(&output.stderr)
                        .lines()
                        .last()
                        .unwrap_or("")
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n  {}", failures.join("\n  "));
}

/// Runs the case its parent names, and nothing otherwise.
#[test]
fn child() {
    let Ok(case) = std::env::var(CASE) else {
        return;
    };
    let fields: Vec<usize> = case.split(' ').map(|x| x.parse().unwrap()).collect();
    let [operation, n, bytes] = fields[..] else {
        panic!("a malformed case: {case}");
    };
    std::thread::Builder::new()
        .stack_size(bytes)
        .spawn(move || dispatch(operation, n))
        .unwrap()
        .join()
        .unwrap();
}
