//! Longer chains: three and five coordinates through the engine, written
//! from positions alone.
//!
//! Three coordinates take the last of the fixed-size linear solves and five
//! the general one, here with a positive definite mass matrix. Everything
//! runs without an allocator.

mod common;

use common::{Lcg, Pendulum, advance, check, mismatch, rel, tight};
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

/// `N` point masses on massless rods, each hanging from the one before.
/// Angles are measured from straight down.
struct Chain<const N: usize> {
    masses: [f64; N],
    lengths: [f64; N],
    g: f64,
}

impl<const N: usize> Chain<N> {
    /// Where bob `k` is. No velocity is written anywhere in this file.
    fn bob<S: Scalar>(&self, k: usize, q: [S; N]) -> [S; 2] {
        let (mut x, mut y) = (S::ZERO, S::ZERO);
        for i in 0..=k {
            let (sin, cos) = q[i].sin_cos();
            x += sin * self.lengths[i];
            y -= cos * self.lengths[i];
        }
        [x, y]
    }

    fn at_rest_energy_scale(&self) -> f64 {
        self.g * self.masses.iter().sum::<f64>() * self.lengths.iter().sum::<f64>()
    }
}

impl<const N: usize> Lagrangian<N> for Chain<N> {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> S {
        let mut total = S::ZERO;
        for k in 0..N {
            let (pos, vel) = velocity_of(|q, _t| self.bob(k, q), q, dq, t);
            total += (vel[0].powi(2) + vel[1].powi(2)) * (0.5 * self.masses[k]);
            total -= pos[1] * (self.masses[k] * self.g);
        }
        total
    }
}

fn uneven<const N: usize>() -> Chain<N> {
    Chain {
        masses: core::array::from_fn(|i| 1.0 + 0.3 * i as f64),
        lengths: core::array::from_fn(|i| 1.0 - 0.1 * i as f64),
        g: 9.81,
    }
}

#[test]
fn one_and_two_links_are_the_pendulum_and_the_double_pendulum() {
    // The same physics written two ways: positions differentiated by the
    // crate, against kinetic energies expanded by hand.
    let mut rng = Lcg(51);
    let single = Chain {
        masses: [1.0],
        lengths: [0.7],
        g: 9.81,
    };
    let double = Chain {
        masses: [1.3, 0.7],
        lengths: [0.9, 1.4],
        g: 9.81,
    };
    let by_hand = DoublePendulum {
        m1: 1.3,
        m2: 0.7,
        l1: 0.9,
        l2: 1.4,
        g: 9.81,
    };
    let mut worst = 0.0_f64;
    for _ in 0..40 {
        let (q, dq) = ([rng.range(-3.0, 3.0)], [rng.range(-2.0, 2.0)]);
        let a = accelerations(&single, q, dq, 0.0).unwrap();
        let b = accelerations(&Pendulum { l: 0.7, g: 9.81 }, q, dq, 0.0).unwrap();
        worst = worst.max(rel(a[0], b[0]));

        let q = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let dq = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];
        let a = accelerations(&double, q, dq, 0.0).unwrap();
        let b = accelerations(&by_hand, q, dq, 0.0).unwrap();
        worst = worst.max(rel(a[0], b[0])).max(rel(a[1], b[1]));
        worst = worst.max(rel(
            energy(&double, q, dq, 0.0).total,
            energy(&by_hand, q, dq, 0.0).total,
        ));
    }
    check(
        "chains of 1 and 2 links vs hand-expanded Lagrangians",
        worst,
        3e-14,
    );
}

fn oracle_sweep<const N: usize>(seed: u64) -> f64 {
    let chain = uneven::<N>();
    let mut rng = Lcg(seed);
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let q: [f64; N] = core::array::from_fn(|_| rng.range(-3.0, 3.0));
        let dq: [f64; N] = core::array::from_fn(|_| rng.range(-2.0, 2.0));
        worst = worst.max(mismatch(&chain, q, dq, 0.0));
        worst = worst.max(mismatch(&chain, q, [0.0; N], 0.0));
    }
    worst
}

#[test]
fn longer_chains_match_the_hyperdual_oracle() {
    check("oracle: chain of 3", oracle_sweep::<3>(53), 2e-13);
    check("oracle: chain of 5", oracle_sweep::<5>(55), 3e-13);
    check("oracle: chain of 8", oracle_sweep::<8>(58), 2e-12);
}

/// Relative energy drift at the knots of a run from rest with every rod
/// raised to the same angle.
fn drift<const N: usize>(angle: f64, t_end: f64, config: SimConfig) -> f64 {
    let chain = uneven::<N>();
    let q0 = [angle; N];
    let e0 = energy(&chain, q0, [0.0; N], 0.0).total;
    let mut stepper = Stepper::new(&chain, 0.0, q0, [0.0; N], config).unwrap();
    let mut worst = 0.0_f64;
    while stepper.state().t < t_end {
        stepper.step_bounded(t_end).unwrap();
        let s = stepper.state();
        worst = worst.max((energy(&chain, s.q, s.dq, s.t).total - e0).abs());
    }
    worst / chain.at_rest_energy_scale()
}

#[test]
fn longer_chains_conserve_energy() {
    // Raised to horizontal and let go: every rod whips. The runs are short
    // because five coordinates through nested dual numbers are slow in a
    // debug build, and this file is compiled in one for every feature set.
    let level = core::f64::consts::FRAC_PI_2;
    check(
        "chain of 3, energy, dopri5 (tight), t in [0, 3]",
        drift::<3>(level, 3.0, tight()),
        1e-12,
    );
    check(
        "chain of 5, energy, dopri5 (tight), t in [0, 3]",
        drift::<5>(level, 3.0, tight()),
        1e-12,
    );
    check(
        "chain of 3, energy band, variational4 dt=0.002, 400 steps",
        drift::<3>(level, 0.8, SimConfig::variational4(0.002)),
        1.5e-8,
    );
    check(
        "chain of 5, energy band, variational4 dt=0.002, 400 steps",
        drift::<5>(level, 0.8, SimConfig::variational4(0.002)),
        1.5e-8,
    );
}

fn hanging_stiffness<const N: usize>() -> f64 {
    // About the hanging equilibrium M·δq̈ = −K·δq, where K is the Hessian of
    // the potential: diagonal, K_ii = g·l_i·(mass hanging from rod i).
    let chain = uneven::<N>();
    let lin = lagrangian::linearize_free(&chain, [0.0; N], [0.0; N], 0.0).unwrap();
    let mass = mass_matrix(&chain, [0.0; N], [0.0; N], 0.0);
    let mut worst = 0.0_f64;
    for i in 0..N {
        let below: f64 = chain.masses[i..].iter().sum();
        for j in 0..N {
            let m_a: f64 = (0..N).map(|k| mass[i][k] * lin.a_q[k][j]).sum();
            let expected = if i == j {
                -chain.g * chain.lengths[i] * below
            } else {
                0.0
            };
            worst = worst.max((m_a - expected).abs() / (chain.g * below));
            assert_eq!(
                lin.a_dq[i][j], 0.0,
                "no damping, no velocity coupling at rest"
            );
        }
    }
    worst
}

#[test]
fn a_hanging_chain_linearizes_to_its_stiffness_matrix() {
    check(
        "chain of 3: M·a_q vs −K at the hanging equilibrium",
        hanging_stiffness::<3>(),
        2e-14,
    );
    check(
        "chain of 5: M·a_q vs −K at the hanging equilibrium",
        hanging_stiffness::<5>(),
        2e-14,
    );
}

#[test]
fn a_chain_hanging_straight_stays_there() {
    let chain = uneven::<5>();
    assert_eq!(
        accelerations(&chain, [0.0; 5], [0.0; 5], 0.0).unwrap(),
        [0.0; 5]
    );
    let end = advance(&chain, [0.0; 5], [0.0; 5], 3.0, SimConfig::default());
    assert_eq!((end.q, end.dq), ([0.0; 5], [0.0; 5]));
    let end = advance(
        &chain,
        [0.0; 5],
        [0.0; 5],
        3.0,
        SimConfig::variational4(0.01),
    );
    assert_eq!((end.q, end.dq), ([0.0; 5], [0.0; 5]));
}
