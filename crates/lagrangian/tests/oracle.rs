//! The engine against an independent oracle.
//!
//! The engine extracts every term of the equations of motion in `N + 2`
//! passes of `Grad<N>` and `Dual<Grad<N>>`. Here the same terms are computed
//! one scalar at a time with `HyperDual`, a different type on a different
//! code path, for every built-in system, at moving states and at rest.

mod common;

use common::{CoupledOscillators, Lcg, Pendulum, check, mismatch, rel};
use lagrangian::prelude::*;
use lagrangian::systems::*;
use lagrangian::velocity_from_momentum;

/// The worst mismatch over random moving states and one state at rest.
/// `state` draws coordinates the system is regular at.
fn sweep<L: Lagrangian<N>, const N: usize>(
    sys: &L,
    seed: u64,
    state: impl Fn(&mut Lcg) -> [f64; N],
) -> f64 {
    let mut rng = Lcg(seed);
    let mut worst = 0.0_f64;
    for _ in 0..40 {
        let q = state(&mut rng);
        let dq = core::array::from_fn(|_| rng.range(-2.0, 2.0));
        let t = rng.range(0.0, 5.0);
        worst = worst.max(mismatch(sys, q, dq, t));
        // At rest: the state every simulation starts from, and the one the
        // closed-form derivative of `powi` turns into NaN.
        worst = worst.max(mismatch(sys, q, [0.0; N], t));
        let ddq = accelerations(sys, q, [0.0; N], t).expect("regular at rest");
        assert!(ddq.iter().all(|x| x.is_finite()));
    }
    worst
}

fn angles<const N: usize>(rng: &mut Lcg) -> [f64; N] {
    core::array::from_fn(|_| rng.range(-3.0, 3.0))
}

#[test]
fn the_engine_matches_the_hyperdual_oracle_on_every_system() {
    let limit = 1e-13;
    check(
        "oracle: double pendulum",
        sweep(
            &DoublePendulum {
                m1: 1.3,
                m2: 0.7,
                l1: 0.9,
                l2: 1.4,
                g: 9.81,
            },
            1,
            angles,
        ),
        limit,
    );
    check(
        "oracle: bead on rotating hoop",
        sweep(&BeadOnRotatingHoop::default(), 2, angles),
        limit,
    );
    // Explicitly time dependent: this is the system that exercises ∂/∂t.
    check(
        "oracle: Kapitza pendulum",
        sweep(
            &KapitzaPendulum {
                amplitude: 0.2,
                frequency: 7.0,
                ..Default::default()
            },
            3,
            angles,
        ),
        limit,
    );
    check(
        "oracle: cart-pole with damping",
        sweep(
            &CartPole {
                cart_damping: 0.3,
                pole_damping: 0.05,
                ..Default::default()
            },
            4,
            angles,
        ),
        limit,
    );
    check(
        "oracle: Kepler orbit",
        sweep(&KeplerOrbit { mu: 2.5, m: 0.8 }, 5, |rng| {
            [rng.range(0.5, 4.0), rng.range(-3.0, 3.0)]
        }),
        limit,
    );
    check(
        "oracle: Schwarzschild geodesic",
        sweep(&SchwarzschildGeodesic { rs: 1.0 }, 6, |rng| {
            // Outside the horizon and away from the poles.
            [
                rng.range(-5.0, 5.0),
                rng.range(1.5, 20.0),
                rng.range(0.4, 2.7),
                rng.range(-3.0, 3.0),
            ]
        }),
        // 1/(1 − r_s/r) is the least well conditioned term in the suite.
        3e-13,
    );
    check(
        "oracle: pendulum",
        sweep(&Pendulum { l: 0.7, g: 9.81 }, 7, angles),
        limit,
    );
    check(
        "oracle: coupled oscillators",
        sweep(&CoupledOscillators { k: 1.0, kc: 0.5 }, 8, angles),
        limit,
    );
}

#[test]
fn the_convective_vector_is_the_centrifugal_and_coriolis_terms() {
    // Kepler in polar coordinates, by hand: p = (m ṙ, m r² θ̇), and the
    // total time derivative of p is C + M q̈, so C = (0, 2 m r ṙ θ̇).
    let orbit = KeplerOrbit { mu: 1.0, m: 2.0 };
    let (r, dr, dth) = (1.7, 0.4, -0.9);
    let terms = eom_terms(&orbit, [r, 0.3], [dr, dth], 0.0);
    assert!(rel(terms.convective[0], 0.0) <= 1e-15);
    assert!(rel(terms.convective[1], 2.0 * 2.0 * r * dr * dth) <= 1e-15);
    assert!(rel(terms.grad_q[0], 2.0 * r * dth * dth - 2.0 / (r * r)) <= 1e-15);
    assert_eq!(terms.grad_q[1], 0.0, "θ is cyclic");
    assert_eq!(terms.mass, [[2.0, 0.0], [0.0, 2.0 * r * r]]);
    // r̈ = r θ̇² − μ/r²,  θ̈ = −2 ṙ θ̇ / r
    let ddq = accelerations(&orbit, [r, 0.3], [dr, dth], 0.0).unwrap();
    assert!(rel(ddq[0], r * dth * dth - 1.0 / (r * r)) <= 1e-15);
    assert!(rel(ddq[1], -2.0 * dr * dth / r) <= 1e-15);
}

#[test]
fn the_legendre_split_recovers_kinetic_and_potential_energy() {
    // A natural system, L = T(q, q̇) − V(q) with T quadratic in q̇: the split
    // must give back exactly that T and that V.
    let dp = DoublePendulum {
        m1: 1.3,
        m2: 0.7,
        l1: 0.9,
        l2: 1.4,
        g: 9.81,
    };
    let mut rng = Lcg(11);
    let mut worst = 0.0_f64;
    for _ in 0..40 {
        let q: [f64; 2] = angles(&mut rng);
        let dq = [rng.range(-3.0, 3.0), rng.range(-3.0, 3.0)];
        let e = energy(&dp, q, dq, 0.0);
        let m = mass_matrix(&dp, q, dq, 0.0);
        let kinetic = 0.5
            * (m[0][0] * dq[0] * dq[0] + 2.0 * m[0][1] * dq[0] * dq[1] + m[1][1] * dq[1] * dq[1]);
        let y1 = -dp.l1 * q[0].cos();
        let y2 = y1 - dp.l2 * q[1].cos();
        let potential = dp.g * (dp.m1 * y1 + dp.m2 * y2);
        worst = worst
            .max(rel(e.kinetic, kinetic))
            .max(rel(e.potential, potential))
            .max(rel(e.total, kinetic + potential));
    }
    check(
        "legendre split vs T and V of a natural system",
        worst,
        3e-14,
    );
}

#[test]
fn the_jacobi_integral_is_not_the_naive_energy_in_a_rotating_frame() {
    // The hoop's Lagrangian has a kinetic term with no velocity in it. The
    // conserved quantity subtracts it; ½ q̇ᵀ M q̇ + V adds nothing for it.
    let hoop = BeadOnRotatingHoop::default();
    let (theta, w) = (1.1_f64, 0.7);
    let e = energy(&hoop, [theta], [w], 0.0);
    let r2 = hoop.radius * hoop.radius;
    let spin = 0.5 * hoop.m * r2 * hoop.omega * hoop.omega * theta.sin().powi(2);
    let gravity = -hoop.m * hoop.g * hoop.radius * theta.cos();
    assert!(rel(e.total, 0.5 * hoop.m * r2 * w * w - spin + gravity) <= 1e-14);
    assert!(rel(e.potential, gravity - spin) <= 1e-14);
    assert!(rel(e.kinetic, 0.5 * hoop.m * r2 * w * w) <= 1e-14);
}

#[test]
fn a_velocity_is_recovered_from_its_momentum() {
    let mut rng = Lcg(13);
    let dp = DoublePendulum::default();
    let bh = SchwarzschildGeodesic::default();
    let mut worst = 0.0_f64;
    for _ in 0..20 {
        let (q, dq): ([f64; 2], [f64; 2]) = (angles(&mut rng), angles(&mut rng));
        let p = momentum(&dp, q, dq, 0.0);
        // From a guess that is nowhere near.
        let back = velocity_from_momentum(&dp, q, p, 0.0, [0.0; 2]).unwrap();
        worst = worst.max(rel(back[0], dq[0])).max(rel(back[1], dq[1]));

        // An indefinite mass matrix inverts just the same.
        let q = [0.0, rng.range(2.0, 10.0), rng.range(0.5, 2.5), 0.0];
        let dq: [f64; 4] = angles(&mut rng);
        let p = momentum(&bh, q, dq, 0.0);
        let back = velocity_from_momentum(&bh, q, p, 0.0, [0.0; 4]).unwrap();
        for i in 0..4 {
            worst = worst.max(rel(back[i], dq[i]));
        }
    }
    check("velocity recovered from momentum", worst, 1e-14);
}

#[test]
fn a_degenerate_lagrangian_reports_a_singular_mass_matrix() {
    /// No kinetic term for the second coordinate.
    struct Degenerate;
    impl Lagrangian<2> for Degenerate {
        fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
            dq[0].powi(2) * 0.5 - q[1].powi(2)
        }
    }
    assert_eq!(
        accelerations(&Degenerate, [0.1, 0.2], [0.0, 0.0], 3.0),
        Err(SimError::SingularMassMatrix { t: 3.0 })
    );
    // On the axis sin θ = 0 and the metric loses its φφ entry.
    let bh = SchwarzschildGeodesic { rs: 1.0 };
    assert_eq!(
        accelerations(&bh, [0.0, 5.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.1], 0.0),
        Err(SimError::SingularMassMatrix { t: 0.0 })
    );
    // On the horizon it has an infinite entry instead, which is a state
    // that is not finite rather than a matrix that is singular.
    assert_eq!(
        accelerations(&bh, [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.1], 0.0),
        Err(SimError::NonFiniteState { t: 0.0 })
    );
}

#[test]
fn an_unguarded_square_root_at_zero_is_an_error_not_a_number() {
    /// `|q|` written the way the documentation says not to.
    struct Kink;
    impl Lagrangian<1> for Kink {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5 - (q[0] * q[0]).sqrt()
        }
    }
    assert_eq!(
        accelerations(&Kink, [0.0], [0.0], 1.5),
        Err(SimError::NonFiniteState { t: 1.5 })
    );
    /// The same potential with the guarded function.
    struct Guarded;
    impl Lagrangian<1> for Guarded {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5 - q[0].hypot(S::ZERO)
        }
    }
    assert_eq!(accelerations(&Guarded, [0.0], [0.0], 1.5), Ok([0.0]));
    assert_eq!(accelerations(&Guarded, [0.5], [0.0], 1.5), Ok([-1.0]));
}

#[test]
fn a_cartesian_force_becomes_a_generalized_force() {
    /// A pendulum with linear drag on the bob, written in Cartesian terms
    /// and projected by virtual work: Q = F · ∂r/∂θ.
    struct Dragged {
        l: f64,
        c: f64,
    }
    impl Lagrangian<1> for Dragged {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            (dq[0] * self.l).powi(2) * 0.5 + q[0].cos() * (9.81 * self.l)
        }
        fn generalized_force<S: Scalar>(&self, q: [S; 1], dq: [S; 1], t: S) -> [S; 1] {
            let (_, vel) = velocity_of(
                |q, _t| [q[0].sin() * self.l, -q[0].cos() * self.l],
                q,
                dq,
                t,
            );
            // ∂r/∂θ is the velocity per unit θ̇.
            let (sin, cos) = q[0].sin_cos();
            let tangent = [cos * self.l, sin * self.l];
            [(vel[0] * tangent[0] + vel[1] * tangent[1]) * -self.c]
        }
    }
    let sys = Dragged { l: 1.5, c: 0.4 };
    let (theta, w) = (0.8_f64, -1.3);
    let ddq = accelerations(&sys, [theta], [w], 0.0).unwrap();
    // l² θ̈ = −g l sin θ − c l² θ̇
    let expected = -9.81 / sys.l * theta.sin() - sys.c * w;
    assert!(rel(ddq[0], expected) <= 1e-14);
}
