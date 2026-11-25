//! The built-in systems against the physics they are named for.
#![cfg(feature = "alloc")]

mod common;

use core::f64::consts::{FRAC_PI_2, PI};

use common::{check, rel, tight};
use lagrangian::prelude::*;
use lagrangian::systems::*;

/// A Kapitza pendulum whose drive puts `a²ν²/(2gl)` at `ratio`.
fn shaken(ratio: f64) -> KapitzaPendulum {
    let (length, g, amplitude) = (1.0_f64, 9.81_f64, 0.05_f64);
    let pendulum = KapitzaPendulum {
        m: 1.0,
        length,
        g,
        amplitude,
        frequency: (2.0 * ratio * g * length).sqrt() / amplitude,
    };
    assert!(rel(pendulum.stability_ratio(), ratio) < 1e-14);
    pendulum
}

/// The largest departure from upright over `periods` of the drive, starting
/// `0.08 rad` away from it. Stops early once the pendulum has plainly
/// fallen.
fn departure(pendulum: &KapitzaPendulum, periods: f64) -> f64 {
    let t_end = periods * 2.0 * PI / pendulum.frequency;
    let mut stepper =
        Stepper::new(pendulum, 0.0, [PI + 0.08], [0.0], SimConfig::default()).unwrap();
    let mut worst = 0.0_f64;
    while stepper.state().t < t_end && worst < 2.0 {
        let segment = stepper.step_bounded(t_end).unwrap();
        worst = worst.max((segment.end().q[0] - PI).abs());
    }
    worst
}

#[test]
fn a_fast_enough_drive_holds_the_pendulum_upside_down() {
    // Twice the stability bound a²ν² > 2gl: the pendulum oscillates about
    // straight up and never gets further from it than where it started.
    // A bound on where it goes, not an error to shrink: it starts 0.08 rad
    // from upright and must stay within 0.1.
    let stable = departure(&shaken(2.0), 500.0);
    println!("measured  Kapitza at ratio 2: departure {stable:.3} rad over 500 drive periods");
    assert!(stable < 0.1, "departure {stable}");

    // Half the bound, same start: it falls.
    let fallen = departure(&shaken(0.5), 500.0);
    println!("measured  Kapitza at ratio 0.5: departure {fallen:.2} rad (it falls)");
    assert!(fallen > 1.0);

    // And with no drive at all upright is simply unstable.
    let unshaken = KapitzaPendulum {
        amplitude: 0.0,
        ..shaken(2.0)
    };
    assert!(departure(&unshaken, 500.0) > 1.0);
}

#[test]
fn the_shaken_pivot_enters_through_the_time_derivative() {
    // Hanging straight down and at rest, the only acceleration comes from
    // the pivot. Vertical shaking cannot swing a pendulum that hangs
    // exactly vertically, but tilted it feels gravity plus the pivot's
    // acceleration: θ̈ = −(g + ÿ_pivot)/l · sin θ.
    let p = KapitzaPendulum {
        amplitude: 0.2,
        frequency: 7.0,
        ..Default::default()
    };
    assert_eq!(accelerations(&p, [0.0], [0.0], 0.3).unwrap(), [0.0]);
    let (theta, t) = (0.6_f64, 0.37_f64);
    let pivot_acc = -p.amplitude * p.frequency * p.frequency * (p.frequency * t).cos();
    let expected = -(p.g + pivot_acc) / p.length * theta.sin();
    let got = accelerations(&p, [theta], [0.0], t).unwrap()[0];
    check(
        "Kapitza: acceleration vs g + pivot acceleration",
        rel(got, expected),
        1e-14,
    );
}

/// A bound, strongly precessing orbit: from `r = 10 r_s` with less angular
/// velocity than a circular orbit needs, so that is the far point.
fn precessing_orbit() -> (SchwarzschildGeodesic, [f64; 4], [f64; 4]) {
    let hole = SchwarzschildGeodesic { rs: 1.0 };
    (
        hole,
        [0.0, 10.0, FRAC_PI_2, 0.0],
        hole.timelike_velocity(10.0, 0.0, 0.02),
    )
}

#[test]
fn a_relativistic_orbit_conserves_what_the_symmetries_say() {
    let (hole, q0, u0) = precessing_orbit();
    let traj = simulate(&hole, (0.0, 2000.0), q0, u0, &tight()).unwrap();

    // The metric does not depend on t or φ: their momenta are constants of
    // the motion (minus the energy and the angular momentum per unit mass),
    // and so is the length of the four-velocity, −1 for a massive particle.
    let p0 = momentum(&hole, q0, u0, 0.0);
    let norm0 = 2.0 * hole.lagrangian(q0, u0, 0.0);
    assert!(
        rel(norm0, -1.0) < 1e-14,
        "the four-velocity is normalized: {norm0}"
    );
    let (mut p_t, mut p_phi, mut norm, mut tilt) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut r_min, mut r_max) = (f64::INFINITY, 0.0_f64);
    for s in traj.knots() {
        let p = momentum(&hole, s.q, s.dq, s.t);
        p_t = p_t.max(rel(p[0], p0[0]));
        p_phi = p_phi.max(rel(p[3], p0[3]));
        norm = norm.max(rel(2.0 * hole.lagrangian(s.q, s.dq, s.t), norm0));
        tilt = tilt.max((s.q[2] - FRAC_PI_2).abs());
        r_min = r_min.min(s.q[1]);
        r_max = r_max.max(s.q[1]);
    }
    check(
        "Schwarzschild: energy p_t along a precessing orbit (tight)",
        p_t,
        1e-11,
    );
    check(
        "Schwarzschild: angular momentum p_phi (tight)",
        p_phi,
        4e-11,
    );
    check("Schwarzschild: four-velocity norm (tight)", norm, 5e-12);
    check(
        "Schwarzschild: departure from the equatorial plane",
        tilt,
        1e-14,
    );

    // Bound, and well inside the strong field.
    println!("measured  Schwarzschild orbit: r in [{r_min:.3}, {r_max:.3}] r_s");
    assert!(r_max <= 10.0 + 1e-9 && r_min > 3.0 && r_min < 5.0);

    // The orbit does not close: each closest approach comes later in angle
    // than a Newtonian ellipse's would, by far more than a rounding error.
    let closest = traj.crossings(|s| s.dq[1], CrossingDirection::Rising);
    assert!(closest.len() >= 5, "{} closest approaches", closest.len());
    let advance = closest[2].q[3] - closest[1].q[3] - 2.0 * PI;
    println!("measured  Schwarzschild orbit: periapsis advances {advance:.4} rad per orbit");
    assert!(advance > 1.0, "strong-field precession");
    // …by the same amount every time: the motion is periodic in r.
    let again = closest[3].q[3] - closest[2].q[3] - 2.0 * PI;
    check(
        "Schwarzschild: periapsis advance is the same each orbit",
        rel(advance, again),
        1e-11,
    );
}

#[test]
fn the_indefinite_metric_goes_through_the_general_solver() {
    // Four coordinates, so not one of the fixed-size paths; one negative
    // and three positive eigenvalues, so not Cholesky territory.
    let (hole, q0, u0) = precessing_orbit();
    let g = mass_matrix(&hole, q0, u0, 0.0);
    let f = 1.0 - hole.rs / q0[1];
    assert!(rel(g[0][0], -f) < 1e-15 && rel(g[1][1], 1.0 / f) < 1e-15);
    assert!(rel(g[2][2], 100.0) < 1e-15 && rel(g[3][3], 100.0) < 1e-15);
    assert!(g[0][0] < 0.0 && g[1][1] > 0.0);
    // The geodesic equation for r at a turning point, by hand:
    // r̈ = −(r_s/2r²)·f·ṫ² + r·f·φ̇² (with ṙ = 0).
    let ddq = accelerations(&hole, q0, u0, 0.0).unwrap();
    let r = q0[1];
    let expected = -hole.rs / (2.0 * r * r) * f * u0[0] * u0[0] + r * f * u0[3] * u0[3];
    check(
        "Schwarzschild: radial acceleration vs geodesic equation",
        rel(ddq[1], expected),
        1e-14,
    );
}

#[test]
fn the_metric_just_outside_the_horizon_is_not_singular() {
    // g_tt = −f and g_rr = 1/f with f → 0 at the horizon: 0.1.0 called the
    // metric singular from r = r_s·(1 + 1e-7) inward, although it is
    // invertible everywhere outside the horizon (#16). Held at rest there,
    // r̈ = −(r_s/2r²)·f·ṫ² = −r_s/(2r²), since f·ṫ² = 1.
    let hole = SchwarzschildGeodesic::default();
    let mut worst = 0.0_f64;
    for gap in [1e-6, 1e-7, 1e-9, 1e-12] {
        let r = hole.rs * (1.0 + gap);
        let q = [0.0, r, FRAC_PI_2, 0.0];
        let u = hole.timelike_velocity(r, 0.0, 0.0);
        let ddq = accelerations(&hole, q, u, 0.0)
            .unwrap_or_else(|e| panic!("r = r_s·(1 + {gap:e}): {e}"));
        worst = worst.max(rel(ddq[1], -hole.rs / (2.0 * r * r)));
        assert_eq!((ddq[0], ddq[2], ddq[3]), (0.0, 0.0, 0.0), "{ddq:?}");
    }
    check(
        "Schwarzschild: r̈ at rest just outside the horizon",
        worst,
        LIMIT_HORIZON,
    );
}

/// Measured 5.6e-17, one rounding of `r_s/(2r²)` (PLAN.md §7.1).
const LIMIT_HORIZON: f64 = 6e-16;

#[test]
fn light_circles_the_photon_sphere_and_only_just() {
    let hole = SchwarzschildGeodesic { rs: 1.0 };
    let radius = hole.photon_sphere();
    assert_eq!(radius, 1.5);
    let start = |r: f64| ([0.0, r, FRAC_PI_2, 0.0], hole.null_velocity(r, 0.0, 1.0));

    // On the sphere a tangential ray has no radial acceleration, and its
    // tangent is null.
    let (q0, k0) = start(radius);
    assert!(hole.lagrangian(q0, k0, 0.0).abs() < 1e-15, "a null ray");
    let radial = accelerations(&hole, q0, k0, 0.0).unwrap()[1];
    check(
        "photon sphere: radial acceleration of a tangential ray",
        radial.abs(),
        2e-15,
    );

    // A millionth of r_s inside it falls in; a millionth outside escapes.
    let fate = |r: f64| {
        let (q0, k0) = start(r);
        let mut stepper = Stepper::new(&hole, 0.0, q0, k0, tight()).unwrap();
        while stepper.state().t < 60.0 {
            stepper.step_bounded(60.0).unwrap();
            let now = stepper.state().q[1];
            if !(1.1..=4.0).contains(&now) {
                return now;
            }
        }
        stepper.state().q[1]
    };
    let (inside, outside) = (fate(radius - 1e-6), fate(radius + 1e-6));
    println!("measured  photon sphere ± 1e-6: ends at r = {inside:.3} and r = {outside:.3}");
    assert!(inside < 1.1, "the ray started inside spirals in");
    assert!(outside > 4.0, "the ray started outside escapes");
}

#[test]
fn scene_maps_give_the_velocity_and_acceleration_of_real_points() {
    // A Kepler orbit in polar coordinates, looked at in the plane: the
    // acceleration of the body must be Newton's, −μ·r̂/r², although nothing
    // Cartesian was ever written down for it.
    let orbit = KeplerOrbit { mu: 1.7, m: 0.6 };
    let traj = simulate(&orbit, (0.0, 20.0), [1.0, 0.0], [0.0, 1.2], &tight()).unwrap();
    let mut worst = 0.0_f64;
    for k in 0..=200 {
        let t = 0.1 * k as f64;
        let [body] = traj.kinematics(&orbit, t);
        let r = body.pos[0].hypot(body.pos[1]);
        for i in 0..2 {
            worst = worst.max(rel(body.acc[i], -orbit.mu * body.pos[i] / (r * r * r)));
        }
        // And the velocity is tangent to the path it traces.
        let h = 1e-6;
        let ([ahead], [behind]) = (
            traj.kinematics(&orbit, t + h),
            traj.kinematics(&orbit, t - h),
        );
        if k > 0 && k < 200 {
            for i in 0..2 {
                assert!(rel(body.vel[i], (ahead.pos[i] - behind.pos[i]) / (2.0 * h)) < 1e-7);
            }
        }
    }
    // Mid-step samples: the limit is the interpolant's accuracy.
    check(
        "Kepler: Cartesian acceleration vs inverse-square law (tight)",
        worst,
        1e-8,
    );
}

#[test]
fn the_double_pendulum_scene_matches_its_geometry() {
    let dp = DoublePendulum {
        m1: 1.0,
        m2: 1.0,
        l1: 0.8,
        l2: 1.3,
        g: 9.81,
    };
    let state = State {
        t: 0.0,
        q: [0.7, -1.9],
        dq: [1.5, -0.4],
        ddq: [0.0, 0.0],
    };
    let [first, second] = kinematics(&dp, &state);
    // Rod lengths are what the positions say they are.
    assert!(rel(first.pos[0].hypot(first.pos[1]), 0.8) < 1e-15);
    assert!(
        rel(
            (second.pos[0] - first.pos[0]).hypot(second.pos[1] - first.pos[1]),
            1.3
        ) < 1e-15
    );
    // A bob on a rod moves at l·ω, perpendicular to the rod…
    assert!(rel(first.vel[0].hypot(first.vel[1]), 0.8 * 1.5) < 1e-15);
    assert!((first.vel[0] * first.pos[0] + first.vel[1] * first.pos[1]).abs() < 1e-15);
    // …and with no angular acceleration its acceleration is centripetal:
    // l·ω² toward the pivot.
    assert!(rel(first.acc[0].hypot(first.acc[1]), 0.8 * 1.5 * 1.5) < 1e-15);
    assert!(rel(first.acc[0], -first.pos[0] * 1.5 * 1.5) < 1e-15);

    // The kinetic energy the Lagrangian uses is ½ m v² of these very points.
    let e = energy(&dp, state.q, state.dq, 0.0);
    let speed_sq = |k: &Kinematics<2>| k.vel[0] * k.vel[0] + k.vel[1] * k.vel[1];
    assert!(rel(e.kinetic, 0.5 * (speed_sq(&first) + speed_sq(&second))) < 1e-14);
}

#[test]
fn the_cart_pole_falls_the_way_gravity_points() {
    let cart = CartPole::default();
    // Tilted to the right of upright and released: the pole accelerates
    // further right, and the cart, pushed by the hinge, moves left.
    let ddq = accelerations(&cart, [0.0, 0.2], [0.0, 0.0], 0.0).unwrap();
    assert!(ddq[1] > 0.0 && ddq[0] < 0.0);
    // With nothing external acting, x is cyclic: horizontal momentum stays
    // zero while the pole falls and swings underneath.
    let traj = simulate(&cart, (0.0, 5.0), [0.0, 0.2], [0.0, 0.0], &tight()).unwrap();
    let worst = traj
        .knots()
        .map(|s| momentum(&cart, s.q, s.dq, s.t)[0].abs())
        .fold(0.0, f64::max);
    check(
        "cart-pole: horizontal momentum of the free system (tight)",
        worst,
        5e-13,
    );
    // The centre of mass therefore does not move horizontally.
    let [body, tip] = traj.kinematics(&cart, 5.0);
    let centre =
        (cart.m_cart * body.pos[0] + cart.m_pole * tip.pos[0]) / (cart.m_cart + cart.m_pole);
    let start = cart.m_pole * cart.length * 0.2_f64.sin() / (cart.m_cart + cart.m_pole);
    check(
        "cart-pole: drift of the centre of mass (tight)",
        (centre - start).abs(),
        1e-12,
    );
}

#[test]
fn friction_takes_energy_out_at_the_rate_it_does_work() {
    // dE/dt = Q · q̇ for a non-conservative force. Integrated along the
    // trajectory, the energy lost must equal the work done by friction.
    let cart = CartPole {
        cart_damping: 0.4,
        pole_damping: 0.1,
        ..Default::default()
    };
    let traj = simulate(&cart, (0.0, 6.0), [0.0, 0.5], [0.3, 0.0], &tight()).unwrap();
    let lost = traj.energy(&cart, 0.0).total - traj.energy(&cart, 6.0).total;
    assert!(lost > 0.0);
    // Work by five-point Gauss-Legendre on each segment, as `action` does.
    let nodes = [
        0.0,
        -0.538_469_310_105_683_1,
        0.538_469_310_105_683_1,
        -0.906_179_845_938_664,
        0.906_179_845_938_664,
    ];
    let weights = [
        0.568_888_888_888_888_9,
        0.478_628_670_499_366_5,
        0.478_628_670_499_366_5,
        0.236_926_885_056_189_1,
        0.236_926_885_056_189_1,
    ];
    let mut work = 0.0;
    for segment in traj.segments() {
        for (x, w) in nodes.iter().zip(weights) {
            let s = segment.eval(0.5 * (1.0 + x));
            let power =
                cart.cart_damping * s.dq[0] * s.dq[0] + cart.pole_damping * s.dq[1] * s.dq[1];
            work += 0.5 * segment.duration() * w * power;
        }
    }
    check(
        "damped cart-pole: energy lost vs work done by friction (tight)",
        rel(lost, work),
        2e-12,
    );
}
