//! Inputs and linearization.

mod common;

use common::{Lcg, check, rel};
use lagrangian::prelude::*;
use lagrangian::systems::{CartPole, DoublePendulum};
use lagrangian::{Linearization, linearize_free};

/// The state blocks by central differences of the accelerations.
fn by_differences<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> ([[f64; N]; N], [[f64; N]; N]) {
    let h = 1e-6;
    let mut a_q = [[0.0; N]; N];
    let mut a_dq = [[0.0; N]; N];
    for j in 0..N {
        let (mut q_hi, mut q_lo, mut v_hi, mut v_lo) = (q, q, dq, dq);
        q_hi[j] += h;
        q_lo[j] -= h;
        v_hi[j] += h;
        v_lo[j] -= h;
        let dq_hi = accelerations(sys, q_hi, dq, t).unwrap();
        let dq_lo = accelerations(sys, q_lo, dq, t).unwrap();
        let dv_hi = accelerations(sys, q, v_hi, t).unwrap();
        let dv_lo = accelerations(sys, q, v_lo, t).unwrap();
        for i in 0..N {
            a_q[i][j] = (dq_hi[i] - dq_lo[i]) / (2.0 * h);
            a_dq[i][j] = (dv_hi[i] - dv_lo[i]) / (2.0 * h);
        }
    }
    (a_q, a_dq)
}

fn worst_block<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..N {
        for j in 0..N {
            worst = worst.max(rel(a[i][j], b[i][j]));
        }
    }
    worst
}

#[test]
fn the_cart_pole_linearizes_to_its_closed_form() {
    // With θ from upright and the tip at x + l·sin θ, small motions obey
    //   ẍ = (u − m g θ)/M,   θ̈ = ((M + m) g θ − u)/(M l).
    let mut worst = 0.0_f64;
    for (m_cart, m_pole, length, g) in [
        (1.0, 0.1, 0.5, 9.81),
        (2.3, 0.7, 1.4, 1.62),
        (0.4, 1.9, 0.3, 24.8),
    ] {
        let cart = CartPole {
            m_cart,
            m_pole,
            length,
            g,
            ..Default::default()
        };
        let lin = linearize(&cart, [0.7, 0.0], [0.0, 0.0], [0.0], 0.0).unwrap();
        let expected = Linearization {
            a_q: [
                [0.0, -m_pole * g / m_cart],
                [0.0, (m_cart + m_pole) * g / (m_cart * length)],
            ],
            a_dq: [[0.0; 2]; 2],
            b: [[1.0 / m_cart], [-1.0 / (m_cart * length)]],
        };
        worst = worst
            .max(worst_block(&lin.a_q, &expected.a_q))
            .max(worst_block(&lin.a_dq, &expected.a_dq))
            .max(rel(lin.b[0][0], expected.b[0][0]))
            .max(rel(lin.b[1][0], expected.b[1][0]));
    }
    check("cart-pole linearization vs closed form", worst, 1e-14);
}

#[test]
fn linearization_matches_finite_differences_away_from_equilibrium() {
    let mut rng = Lcg(21);
    let dp = DoublePendulum {
        m1: 1.3,
        m2: 0.7,
        l1: 0.9,
        l2: 1.4,
        g: 9.81,
    };
    let cart = CartPole {
        cart_damping: 0.3,
        pole_damping: 0.05,
        ..Default::default()
    };
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let q = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let dq = [rng.range(-2.0, 2.0), rng.range(-2.0, 2.0)];

        let lin = linearize_free(&dp, q, dq, 0.0).unwrap();
        let (a_q, a_dq) = by_differences(&dp, q, dq, 0.0);
        worst = worst
            .max(worst_block(&lin.a_q, &a_q))
            .max(worst_block(&lin.a_dq, &a_dq));

        // A moving, pushed, damped cart: every block is exercised, and the
        // velocity block is where the damping shows up.
        let u = [rng.range(-5.0, 5.0)];
        let lin = linearize(&cart, q, dq, u, 0.0).unwrap();
        let held = WithInput { sys: &cart, u };
        let (a_q, a_dq) = by_differences(&held, q, dq, 0.0);
        worst = worst
            .max(worst_block(&lin.a_q, &a_q))
            .max(worst_block(&lin.a_dq, &a_dq));
        let h = 1e-6;
        let hi = accelerations(
            &WithInput {
                sys: &cart,
                u: [u[0] + h],
            },
            q,
            dq,
            0.0,
        )
        .unwrap();
        let lo = accelerations(
            &WithInput {
                sys: &cart,
                u: [u[0] - h],
            },
            q,
            dq,
            0.0,
        )
        .unwrap();
        for i in 0..2 {
            worst = worst.max(rel(lin.b[i][0], (hi[i] - lo[i]) / (2.0 * h)));
        }
    }
    // The limit is the finite differences' own error, not the engine's.
    check("linearization vs central differences", worst, 3e-8);
}

#[test]
fn an_input_is_a_generalized_force() {
    let cart = CartPole::default();
    let (q, dq) = ([0.2, 0.4], [0.1, -0.3]);
    let free = eom_terms(&cart, q, dq, 0.0);
    let pushed = eom_terms(
        &WithInput {
            sys: &cart,
            u: [3.5],
        },
        q,
        dq,
        0.0,
    );
    assert_eq!(pushed.force, [free.force[0] + 3.5, free.force[1]]);
    assert_eq!(pushed.mass, free.mass);
    assert_eq!(pushed.convective, free.convective);
}

#[test]
fn changing_the_input_between_steps_takes_effect_at_the_knot() {
    let cart = CartPole::default();
    let mut live = Stepper::new(
        WithInput {
            sys: &cart,
            u: [0.0],
        },
        0.0,
        [0.0, 0.1],
        [0.0, 0.0],
        SimConfig::default(),
    )
    .unwrap();
    let before = live.step().unwrap();
    let knot = live.state();

    live.system_mut().u = [4.0];
    let after = live.step().unwrap();

    // Position and velocity do not jump…
    assert_eq!(after.start().q, knot.q);
    assert_eq!(after.start().dq, knot.dq);
    assert_eq!(after.t_start(), before.t_end());
    // …and the acceleration does, to exactly what the pushed system has
    // there. Reusing the previous step's last stage here would be the bug.
    let pushed = accelerations(
        &WithInput {
            sys: &cart,
            u: [4.0],
        },
        knot.q,
        knot.dq,
        knot.t,
    )
    .unwrap();
    assert_eq!(after.start().ddq, pushed);
    assert!((pushed[0] - knot.ddq[0]).abs() > 1.0, "the push is visible");

    // `refresh` shows the new acceleration without stepping.
    let pushed_knot = live.state();
    live.system_mut().u = [0.0];
    assert_eq!(live.state().ddq, pushed_knot.ddq, "stale until refreshed");
    live.refresh().unwrap();
    let s = live.state();
    assert_eq!(
        s.ddq,
        accelerations(
            &WithInput {
                sys: &cart,
                u: [0.0]
            },
            s.q,
            s.dq,
            s.t
        )
        .unwrap()
    );
}

#[test]
fn a_held_input_drives_the_variational_solver_too() {
    // A constant force on a frictionless cart with the pole removed from
    // the picture (θ̇ = 0 is not preserved, so compare total momentum):
    // the cart-pole's total horizontal momentum grows at exactly u.
    let cart = CartPole::default();
    let u = 2.0;
    let mut live = Stepper::new(
        WithInput { sys: &cart, u: [u] },
        0.0,
        [0.0, 0.3],
        [0.0, 0.0],
        SimConfig::variational4(0.01),
    )
    .unwrap();
    for _ in 0..200 {
        live.step().unwrap();
    }
    let s = live.state();
    // x is cyclic, so its momentum obeys ṗ = Q_x = u exactly.
    let p_x = momentum(&cart, s.q, s.dq, s.t)[0];
    check(
        "cart momentum under a constant push (variational4)",
        rel(p_x, u * s.t),
        2e-13,
    );
}

#[cfg(feature = "alloc")]
#[test]
fn the_dense_state_space_matrices_have_the_documented_layout() {
    let cart = CartPole::default();
    let lin = linearize(&cart, [0.0, 0.0], [0.0, 0.0], [0.0], 0.0).unwrap();
    let (a, b) = lin.to_dense();
    assert_eq!((a.len(), b.len()), (16, 4));
    // The top half says d/dt (δq) = δq̇.
    assert_eq!(&a[0..4], &[0.0, 0.0, 1.0, 0.0]);
    assert_eq!(&a[4..8], &[0.0, 0.0, 0.0, 1.0]);
    assert_eq!(&a[8..10], &lin.a_q[0]);
    assert_eq!(&a[10..12], &lin.a_dq[0]);
    assert_eq!(&a[12..14], &lin.a_q[1]);
    assert_eq!(&b[0..2], &[0.0, 0.0]);
    assert_eq!(&b[2..4], &[lin.b[0][0], lin.b[1][0]]);
}
