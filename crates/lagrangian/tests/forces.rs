//! Forces the Lagrangian does not contain, and Lagrangians that are not
//! `T − V`: friction, a drive that depends on time, a magnetic field.
//!
//! Everything here runs without an allocator.

mod common;

use common::{advance, check, rel, tight};
use lagrangian::prelude::*;

/// `ẍ + 2ζω·ẋ + ω²·x = 0`, with the friction as a generalized force.
struct Damped {
    omega: f64,
    zeta: f64,
}

impl Lagrangian<1> for Damped {
    fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
        (dq[0].powi(2) - q[0].powi(2) * (self.omega * self.omega)) * 0.5
    }
    fn generalized_force<S: Scalar>(&self, _q: [S; 1], dq: [S; 1], _t: S) -> [S; 1] {
        [dq[0] * (-2.0 * self.zeta * self.omega)]
    }
}

impl Damped {
    /// Position and velocity from `x = 1` at rest (underdamped).
    fn exact(&self, t: f64) -> (f64, f64) {
        let decay = self.zeta * self.omega;
        let ringing = self.omega * (1.0 - self.zeta * self.zeta).sqrt();
        let envelope = (-decay * t).exp();
        (
            envelope * ((ringing * t).cos() + decay / ringing * (ringing * t).sin()),
            -envelope * self.omega * self.omega / ringing * (ringing * t).sin(),
        )
    }

    fn error(&self, config: SimConfig) -> f64 {
        let end = advance(self, [1.0], [0.0], 6.0, config);
        let (x, v) = self.exact(6.0);
        (end.q[0] - x).abs().max((end.dq[0] - v).abs())
    }
}

#[test]
fn friction_damps_an_oscillator_as_the_closed_form_says() {
    let sys = Damped {
        omega: 3.0,
        zeta: 0.15,
    };
    check(
        "damped oscillator at t = 6, dopri5 (default)",
        sys.error(SimConfig::default()),
        2e-9,
    );
    check(
        "damped oscillator at t = 6, dopri5 (tight)",
        sys.error(tight()),
        2e-12,
    );
    check(
        "damped oscillator at t = 6, variational4 dt=0.01",
        sys.error(SimConfig::variational4(0.01)),
        2e-6,
    );
    check(
        "damped oscillator at t = 6, variational2 dt=0.01",
        sys.error(SimConfig::variational2(0.01)),
        3e-3,
    );
}

#[test]
fn the_variational_solvers_keep_their_order_with_friction() {
    // A velocity-dependent force goes through the implicit solve and its
    // Jacobian. Halving the step must still divide the error by 2^order.
    let sys = Damped {
        omega: 3.0,
        zeta: 0.15,
    };
    let second =
        sys.error(SimConfig::variational2(0.02)) / sys.error(SimConfig::variational2(0.01));
    let fourth =
        sys.error(SimConfig::variational4(0.04)) / sys.error(SimConfig::variational4(0.02));
    println!("measured  convergence ratio with friction, variational2 (expect 4): {second:.3}");
    println!("measured  convergence ratio with friction, variational4 (expect 16): {fourth:.3}");
    assert!((3.6..4.4).contains(&second), "second order: ratio {second}");
    assert!(
        (13.0..19.0).contains(&fourth),
        "fourth order: ratio {fourth}"
    );
}

/// `ẍ + ω₀²·x = F·cos(Ω·t)`, with the drive either as a generalized force
/// or inside the Lagrangian as `x·F·cos(Ω·t)`. The two are the same physics
/// through different parts of the engine.
struct Driven {
    omega0: f64,
    force: f64,
    drive: f64,
    in_lagrangian: bool,
}

impl Driven {
    fn push<S: Scalar>(&self, t: S) -> S {
        (t * self.drive).cos() * self.force
    }

    /// Position and velocity from rest at the origin.
    fn exact(&self, t: f64) -> (f64, f64) {
        let gain = self.force / (self.omega0 * self.omega0 - self.drive * self.drive);
        (
            gain * ((self.drive * t).cos() - (self.omega0 * t).cos()),
            gain * (self.omega0 * (self.omega0 * t).sin() - self.drive * (self.drive * t).sin()),
        )
    }

    fn error(&self, config: SimConfig) -> f64 {
        let end = advance(self, [0.0], [0.0], 9.0, config);
        let (x, v) = self.exact(9.0);
        (end.q[0] - x).abs().max((end.dq[0] - v).abs())
    }
}

impl Lagrangian<1> for Driven {
    fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], t: S) -> S {
        let free = (dq[0].powi(2) - q[0].powi(2) * (self.omega0 * self.omega0)) * 0.5;
        if self.in_lagrangian {
            free + q[0] * self.push(t)
        } else {
            free
        }
    }
    fn generalized_force<S: Scalar>(&self, _q: [S; 1], _dq: [S; 1], t: S) -> [S; 1] {
        if self.in_lagrangian {
            [S::ZERO]
        } else {
            [self.push(t)]
        }
    }
}

#[test]
fn a_drive_that_depends_on_time_is_followed_by_every_solver() {
    for in_lagrangian in [false, true] {
        let sys = Driven {
            omega0: 2.0,
            force: 1.5,
            drive: 3.1,
            in_lagrangian,
        };
        let how = if in_lagrangian { "in L" } else { "as a force" };
        check(
            &format!("driven oscillator at t = 9, drive {how}, dopri5 (tight)"),
            sys.error(tight()),
            6e-12,
        );
        check(
            &format!("driven oscillator at t = 9, drive {how}, variational4 dt=0.01"),
            sys.error(SimConfig::variational4(0.01)),
            7e-7,
        );
        check(
            &format!("driven oscillator at t = 9, drive {how}, variational2 dt=0.01"),
            sys.error(SimConfig::variational2(0.01)),
            2e-3,
        );
    }
}

#[test]
fn a_drive_is_the_same_physics_as_a_force_or_inside_the_lagrangian() {
    let as_force = Driven {
        omega0: 2.0,
        force: 1.5,
        drive: 3.1,
        in_lagrangian: false,
    };
    let inside = Driven {
        in_lagrangian: true,
        ..as_force
    };
    // The same accelerations at the same state…
    let (a, b) = (
        accelerations(&as_force, [0.3], [-0.7], 1.9).unwrap(),
        accelerations(&inside, [0.3], [-0.7], 1.9).unwrap(),
    );
    assert!(rel(a[0], b[0]) <= 1e-15);
    // …but not the same energy: a time-dependent Lagrangian has ∂L/∂t ≠ 0,
    // and the engine reports it.
    assert_eq!(eom_terms(&as_force, [0.3], [-0.7], 1.9).dl_dt, 0.0);
    let expected = -0.3 * 1.5 * 3.1 * (3.1_f64 * 1.9).sin();
    assert!(rel(eom_terms(&inside, [0.3], [-0.7], 1.9).dl_dt, expected) <= 1e-14);

    // And the same trajectory, from every solver.
    let mut worst = 0.0_f64;
    for config in [
        tight(),
        SimConfig::variational4(0.01),
        SimConfig::variational2(0.01),
    ] {
        let (p, q) = (
            advance(&as_force, [0.0], [0.0], 9.0, config),
            advance(&inside, [0.0], [0.0], 9.0, config),
        );
        worst = worst
            .max((p.q[0] - q.q[0]).abs())
            .max((p.dq[0] - q.dq[0]).abs());
    }
    check(
        "driven oscillator: force vs Lagrangian formulation",
        worst,
        2e-14,
    );
}

/// A charge in a uniform magnetic field along z, in the symmetric gauge
/// `A = ½·B × r`, or in the Landau gauge `A = (−B·y, 0)`. The Lagrangian
/// has a term linear in the velocity and no potential energy at all.
struct Cyclotron {
    m: f64,
    charge: f64,
    field: f64,
    landau: bool,
}

impl Lagrangian<2> for Cyclotron {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
        let kinetic = (dq[0].powi(2) + dq[1].powi(2)) * (0.5 * self.m);
        let coupling = if self.landau {
            -(q[1] * dq[0])
        } else {
            (q[0] * dq[1] - q[1] * dq[0]) * 0.5
        };
        kinetic + coupling * (self.charge * self.field)
    }
}

impl Cyclotron {
    fn frequency(&self) -> f64 {
        self.charge * self.field / self.m
    }

    /// From the origin with velocity `(v, 0)`: a circle, clockwise for a
    /// positive charge.
    fn exact(&self, v: f64, t: f64) -> ([f64; 2], [f64; 2]) {
        let w = self.frequency();
        (
            [v / w * (w * t).sin(), v / w * ((w * t).cos() - 1.0)],
            [v * (w * t).cos(), -v * (w * t).sin()],
        )
    }
}

#[test]
fn a_magnetic_field_bends_without_doing_work() {
    let sys = Cyclotron {
        m: 0.7,
        charge: 1.3,
        field: 2.1,
        landau: false,
    };
    let v = 1.9;

    // The force is q·v × B, perpendicular to the velocity.
    let ddq = accelerations(&sys, [0.4, -0.2], [v, 0.6], 0.0).unwrap();
    let w = sys.frequency();
    assert!(rel(ddq[0], w * 0.6) <= 1e-15 && rel(ddq[1], -w * v) <= 1e-15);

    // The canonical momentum is not m·v, and the energy is the kinetic
    // energy alone: ½ q̇ᵀMq̇ + V would agree here, p·q̇ would not.
    let (q, dq) = ([0.4, -0.2], [v, 0.6]);
    let p = momentum(&sys, q, dq, 0.0);
    let half_qb = 0.5 * sys.charge * sys.field;
    assert!(rel(p[0], sys.m * dq[0] - half_qb * q[1]) <= 1e-15);
    assert!(rel(p[1], sys.m * dq[1] + half_qb * q[0]) <= 1e-15);
    let e = energy(&sys, q, dq, 0.0);
    let kinetic = 0.5 * sys.m * (v * v + 0.36);
    assert!(rel(e.total, kinetic) <= 1e-15 && rel(e.kinetic, kinetic) <= 1e-15);
    assert_eq!(e.potential, 0.0);

    // Ten turns of the circle.
    let t_end = 10.0 * 2.0 * core::f64::consts::PI / w;
    let error = |config| {
        let end = advance(&sys, [0.0, 0.0], [v, 0.0], t_end, config);
        let (x, u) = sys.exact(v, t_end);
        let speed = end.dq[0].hypot(end.dq[1]);
        (
            (end.q[0] - x[0]).abs().max((end.q[1] - x[1]).abs()),
            (end.dq[0] - u[0]).abs().max((end.dq[1] - u[1]).abs()),
            (speed - v).abs() / v,
        )
    };
    let (position, velocity, speed) = error(tight());
    check(
        "cyclotron orbit after 10 turns, dopri5 (tight): position",
        position,
        3e-11,
    );
    check(
        "cyclotron orbit after 10 turns, dopri5 (tight): velocity",
        velocity,
        2e-10,
    );
    check(
        "cyclotron orbit after 10 turns, dopri5 (tight): speed",
        speed,
        6e-11,
    );
    let (position, _, speed) = error(SimConfig::variational4(0.005));
    check(
        "cyclotron orbit after 10 turns, variational4 dt=0.005: position",
        position,
        3e-6,
    );
    check(
        "cyclotron orbit after 10 turns, variational4 dt=0.005: speed",
        speed,
        5e-14,
    );
}

#[test]
fn the_motion_does_not_depend_on_the_gauge() {
    // The two vector potentials differ by a gradient, the two Lagrangians
    // by a total time derivative, and the equations of motion by nothing.
    let symmetric = Cyclotron {
        m: 0.7,
        charge: 1.3,
        field: 2.1,
        landau: false,
    };
    let landau = Cyclotron {
        landau: true,
        ..symmetric
    };
    let (q, dq) = ([0.4, -0.2], [1.9, 0.6]);
    let (a, b) = (
        accelerations(&symmetric, q, dq, 0.0).unwrap(),
        accelerations(&landau, q, dq, 0.0).unwrap(),
    );
    assert!(rel(a[0], b[0]) <= 1e-15 && rel(a[1], b[1]) <= 1e-15);
    // The canonical momenta differ; the energy does not.
    assert!(momentum(&symmetric, q, dq, 0.0) != momentum(&landau, q, dq, 0.0));
    assert!(
        rel(
            energy(&symmetric, q, dq, 0.0).total,
            energy(&landau, q, dq, 0.0).total
        ) <= 1e-15
    );
    // In the Landau gauge x is cyclic, so p_x = m·ẋ − q·B·y is conserved.
    let p0 = momentum(&landau, [0.0, 0.0], [1.9, 0.0], 0.0)[0];
    let end = advance(
        &landau,
        [0.0, 0.0],
        [1.9, 0.0],
        25.0,
        SimConfig::variational4(0.005),
    );
    let p1 = momentum(&landau, end.q, end.dq, end.t)[0];
    check(
        "cyclotron, Landau gauge: conserved p_x, variational4, 5000 steps",
        rel(p1, p0),
        5e-15,
    );
}
