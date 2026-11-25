//! Trajectories against closed-form solutions.
#![cfg(feature = "alloc")]

mod common;

use common::{CoupledOscillators, Pendulum, check, tight};
use lagrangian::prelude::*;

/// The largest error in position, at the ends of every step and at three
/// points inside each.
fn oscillator_error(config: &SimConfig) -> f64 {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let traj = simulate(&osc, (0.0, 20.0), [1.0, 0.0], [0.0, 0.0], config).unwrap();
    let mut worst = 0.0_f64;
    for segment in traj.segments() {
        for tau in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let s = segment.eval(tau);
            let exact = osc.exact(s.t);
            worst = worst
                .max((s.q[0] - exact[0]).abs())
                .max((s.q[1] - exact[1]).abs());
        }
    }
    worst
}

#[test]
fn coupled_oscillators_follow_their_normal_modes() {
    check(
        "coupled oscillators vs normal modes, t in [0, 20] (default)",
        oscillator_error(&SimConfig::default()),
        2e-8,
    );
    check(
        "coupled oscillators vs normal modes, t in [0, 20] (tight)",
        oscillator_error(&tight()),
        2e-11,
    );
}

/// The relative error of the first full period of a pendulum released from
/// `2.4 rad`, measured between two upward passages through the bottom.
fn period_error(config: &SimConfig) -> f64 {
    let pendulum = Pendulum { l: 1.0, g: 9.81 };
    let exact = pendulum.period(2.4);
    let traj = simulate(&pendulum, (0.0, 3.0 * exact), [2.4], [0.0], config).unwrap();
    let upward = traj.crossings(|s| s.q[0], CrossingDirection::Rising);
    assert_eq!(upward.len(), 3, "one upward passage per period");
    ((upward[1].t - upward[0].t) - exact).abs() / exact
}

#[test]
fn a_large_swing_has_the_elliptic_integral_period() {
    // Far from the small-angle regime: the period is 1.56 times
    // 2π·sqrt(l/g), so a solver that only got the linear pendulum right
    // would miss by more than half.
    let pendulum = Pendulum { l: 1.0, g: 9.81 };
    let small_angle = 2.0 * core::f64::consts::PI * (1.0_f64 / 9.81).sqrt();
    let ratio = pendulum.period(2.4) / small_angle;
    assert!((1.55..1.57).contains(&ratio), "ratio {ratio}");

    check(
        "pendulum period vs elliptic integral, amplitude 2.4 (default)",
        period_error(&SimConfig::default()),
        2e-8,
    );
    check(
        "pendulum period vs elliptic integral, amplitude 2.4 (tight)",
        period_error(&tight()),
        2e-11,
    );
}

#[test]
fn the_action_of_an_oscillator_matches_its_closed_form() {
    // x = cos t on a unit oscillator: L = ½ẋ² − ½x² = −½cos 2t, so
    // S(T) = −¼ sin 2T.
    struct Unit;
    impl Lagrangian<1> for Unit {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            (dq[0].powi(2) - q[0].powi(2)) * 0.5
        }
    }
    let t_end = 7.3_f64;
    let traj = simulate(&Unit, (0.0, t_end), [1.0], [0.0], &tight()).unwrap();
    let exact = -0.25 * (2.0 * t_end).sin();
    check(
        "action of a unit oscillator vs -sin(2T)/4 (tight)",
        (traj.action(&Unit) - exact).abs(),
        3e-12,
    );
}

#[test]
fn the_true_path_makes_the_action_stationary() {
    // Hamilton's principle, checked numerically: among paths with the same
    // end points, the simulated one has a smaller action than its
    // neighbours (for a short enough interval it is a true minimum).
    struct Falling;
    impl Lagrangian<1> for Falling {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5 - q[0] * 9.81
        }
    }
    // Varied paths y(t) + ε·sin(π t/T), action by the midpoint rule.
    let (t_end, v0) = (1.0_f64, 3.0);
    let action = |eps: f64| {
        let n = 20_000;
        let h = t_end / n as f64;
        let mut sum = 0.0;
        for k in 0..n {
            let t = (k as f64 + 0.5) * h;
            let bump = core::f64::consts::PI * t / t_end;
            let y = v0 * t - 0.5 * 9.81 * t * t + eps * bump.sin();
            let dy = v0 - 9.81 * t + eps * bump.cos() * core::f64::consts::PI / t_end;
            sum += h * Falling.lagrangian([y], [dy], t);
        }
        sum
    };
    let traj = simulate(&Falling, (0.0, t_end), [0.0], [v0], &tight()).unwrap();
    let simulated = traj.action(&Falling);
    assert!((simulated - action(0.0)).abs() < 1e-7);
    for eps in [-0.1, -0.01, 0.01, 0.1] {
        assert!(action(eps) > simulated, "a varied path has more action");
    }
}
