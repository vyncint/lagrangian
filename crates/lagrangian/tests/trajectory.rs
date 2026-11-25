//! The trajectory as a function of time: smoothness, sampling, events.
#![cfg(feature = "alloc")]

mod common;

use common::{CoupledOscillators, Lcg, Pendulum, check, rel, tight};
use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

fn chaotic(config: &SimConfig) -> Trajectory<2> {
    simulate(
        &DoublePendulum::default(),
        (0.0, 20.0),
        [2.2, 2.8],
        [0.0, 0.0],
        config,
    )
    .unwrap()
}

#[test]
fn position_velocity_and_acceleration_are_continuous_across_steps() {
    let traj = chaotic(&SimConfig::dopri5().with_tolerances(1e-10, 1e-12));
    let (mut q, mut dq, mut ddq) = (0.0_f64, 0.0_f64, 0.0_f64);
    for pair in traj.segments().windows(2) {
        let (left, right) = (pair[0].end(), pair[1].start());
        assert_eq!(left.t, right.t);
        for i in 0..2 {
            q = q.max(rel(left.q[i], right.q[i]));
            dq = dq.max(rel(left.dq[i], right.dq[i]));
            ddq = ddq.max(rel(left.ddq[i], right.ddq[i]));
        }
    }
    check("continuity across knots: position", q, 1e-15);
    check("continuity across knots: velocity", dq, 5e-15);
    check("continuity across knots: acceleration", ddq, 3e-13);
}

#[test]
fn the_acceleration_at_a_knot_satisfies_the_equations_of_motion() {
    // Not just continuous: at the start of every segment the stored
    // acceleration is the one the system has at that state, exactly.
    let dp = DoublePendulum::default();
    let traj = chaotic(&SimConfig::default());
    for segment in traj.segments() {
        let s = segment.start();
        assert_eq!(s.ddq, accelerations(&dp, s.q, s.dq, s.t).unwrap());
    }
}

#[test]
fn reported_velocity_and_acceleration_are_derivatives_of_position() {
    let traj = chaotic(&SimConfig::default());
    let h = 1e-5;
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    let mut rng = Lcg(31);
    for _ in 0..200 {
        // Anywhere, step boundaries included: the identity holds across them.
        let t = rng.range(0.1, 19.9);
        let (lo, mid, hi) = (traj.sample(t - h), traj.sample(t), traj.sample(t + h));
        for i in 0..2 {
            first = first.max(rel((hi.q[i] - lo.q[i]) / (2.0 * h), mid.dq[i]));
            second = second.max(rel((hi.dq[i] - lo.dq[i]) / (2.0 * h), mid.ddq[i]));
        }
    }
    // Both limits are the finite differences' own truncation error.
    check("d(position)/dt vs reported velocity", first, 1e-7);
    check("d(velocity)/dt vs reported acceleration", second, 2e-6);
}

#[test]
fn sampling_is_clamped_and_the_ends_are_exact() {
    let traj = chaotic(&SimConfig::default());
    assert_eq!(
        (traj.t_start(), traj.t_end(), traj.duration()),
        (0.0, 20.0, 20.0)
    );
    assert_eq!(traj.start().q, [2.2, 2.8]);
    assert_eq!(traj.sample(-5.0), traj.start());
    assert_eq!(traj.sample(0.0), traj.start());
    assert_eq!(traj.sample(20.0), traj.end());
    assert_eq!(traj.sample(1e9), traj.end());
    assert_eq!(traj.try_sample(-1e-9), None);
    assert_eq!(traj.try_sample(20.0 + 1e-9), None);
    assert_eq!(traj.try_sample(f64::NAN), None);
    assert_eq!(traj.try_sample(7.5), Some(traj.sample(7.5)));
    assert_eq!(traj.sample(7.5).t, 7.5);

    assert_eq!(traj.eval_normalized(0.0), traj.start());
    assert_eq!(traj.eval_normalized(1.0), traj.end());
    assert_eq!(traj.eval_normalized(-3.0), traj.start());
    assert_eq!(traj.eval_normalized(3.0), traj.end());
    assert_eq!(traj.eval_normalized(f64::NAN), traj.start());
    assert_eq!(traj.eval_normalized(0.25), traj.sample(5.0));

    assert_eq!(traj.knots().count(), traj.len_segments() + 1);
    assert_eq!(traj.knots().next(), Some(traj.start()));
    assert_eq!(traj.knots().last(), Some(traj.end()));
}

#[test]
fn a_time_that_is_not_a_number_is_not_a_panic() {
    let traj = chaotic(&SimConfig::default());
    let lost = traj.sample(f64::NAN);
    assert!(lost.t.is_nan() && lost.q[0].is_nan());
    let mut cursor = Cursor::default();
    assert!(traj.sample_with_cursor(f64::NAN, &mut cursor).q[0].is_nan());
    // The cursor is still usable afterwards.
    assert_eq!(traj.sample_with_cursor(3.0, &mut cursor), traj.sample(3.0));
    // An indicator that is NaN everywhere crosses nothing.
    assert!(
        traj.crossings(|_| f64::NAN, CrossingDirection::Either)
            .is_empty()
    );
}

#[test]
fn a_cursor_gives_the_same_answer_as_a_search() {
    let traj = chaotic(&SimConfig::default());
    let mut cursor = Cursor::default();
    // Forward playback, reverse playback, then jumps.
    let forward = (0..=2000).map(|k| k as f64 * 0.01);
    let backward = (0..=2000).rev().map(|k| k as f64 * 0.01);
    let mut rng = Lcg(37);
    let jumps: Vec<f64> = (0..500).map(|_| rng.range(-1.0, 21.0)).collect();
    for t in forward.chain(backward).chain(jumps) {
        assert_eq!(
            traj.sample_with_cursor(t, &mut cursor),
            traj.sample(t),
            "t = {t}"
        );
    }
}

#[test]
fn looping_and_ping_pong_wrap_the_timeline() {
    let traj = chaotic(&SimConfig::default());
    for t in [0.0, 3.7, 19.99] {
        assert_eq!(traj.sample_looped(t), traj.sample(t));
        assert_eq!(traj.sample_ping_pong(t), traj.sample(t));
    }
    let close = |a: State<2>, b: State<2>| rel(a.q[0], b.q[0]) < 1e-9 && rel(a.q[1], b.q[1]) < 1e-9;
    // One lap later, the same place.
    assert!(close(traj.sample_looped(23.7), traj.sample(3.7)));
    assert!(close(traj.sample_looped(-16.3), traj.sample(3.7)));
    assert!(close(traj.sample_looped(403.7), traj.sample(3.7)));
    // Past the end, ping-pong runs backward.
    assert!(close(traj.sample_ping_pong(23.7), traj.sample(16.3)));
    assert!(close(traj.sample_ping_pong(43.7), traj.sample(3.7)));
    assert!(close(traj.sample_ping_pong(-3.7), traj.sample(3.7)));
    // Nothing sensible to wrap to.
    assert_eq!(traj.sample_looped(f64::NAN), traj.start());
    assert_eq!(traj.sample_ping_pong(f64::INFINITY), traj.start());
}

#[test]
fn simulate_is_a_loop_over_the_stepper() {
    // Bit for bit, for every solver: there is no second code path.
    let dp = DoublePendulum::default();
    for config in [
        SimConfig::default(),
        tight(),
        SimConfig::variational4(0.01),
        SimConfig::variational2(0.003),
    ] {
        let traj = simulate(&dp, (0.5, 4.25), [2.2, 2.8], [0.0, 0.0], &config).unwrap();
        let mut stepper = Stepper::new(&dp, 0.5, [2.2, 2.8], [0.0, 0.0], config).unwrap();
        let mut by_hand = Vec::new();
        while stepper.state().t < 4.25 {
            by_hand.push(stepper.step_bounded(4.25).unwrap());
        }
        assert_eq!(traj.segments(), &by_hand[..]);
        assert_eq!(traj.t_end(), 4.25);
    }
}

#[test]
fn a_trajectory_can_be_grown_a_piece_at_a_time() {
    let dp = DoublePendulum::default();
    let whole = simulate(
        &dp,
        (0.0, 3.0),
        [2.2, 2.8],
        [0.0, 0.0],
        &SimConfig::variational4(0.01),
    )
    .unwrap();

    let mut stepper = Stepper::new(
        &dp,
        0.0,
        [2.2, 2.8],
        [0.0, 0.0],
        SimConfig::variational4(0.01),
    )
    .unwrap();
    let mut grown = Trajectory::new(stepper.state());
    assert_eq!((grown.len_segments(), grown.duration()), (0, 0.0));
    assert_eq!(grown.sample(5.0), grown.start());
    for until in [1.0, 2.0, 3.0] {
        grown.extend(&mut stepper, until).unwrap();
        assert_eq!(grown.t_end(), until);
    }
    // Fixed steps that divide the pieces evenly: the same knots either way.
    assert_eq!(grown, whole);
    // Extending to where it already is does nothing.
    grown.extend(&mut stepper, 3.0).unwrap();
    assert_eq!(grown, whole);

    // A stepper that is somewhere else cannot extend it.
    let mut elsewhere =
        Stepper::new(&dp, 9.0, [0.0, 0.0], [0.0, 0.0], SimConfig::default()).unwrap();
    assert!(matches!(
        grown.extend(&mut elsewhere, 10.0),
        Err(SimError::InvalidConfig { .. })
    ));
    // Nor can a segment that does not start at the end be pushed.
    let stray = elsewhere.step().unwrap();
    assert!(matches!(
        grown.push(stray),
        Err(SimError::InvalidConfig { .. })
    ));
    assert_eq!(grown, whole);
}

#[test]
fn the_step_budget_stops_a_run_and_keeps_what_was_done() {
    let dp = DoublePendulum::default();
    let config = SimConfig::variational2(0.01).with_max_steps(25);
    let result = simulate(&dp, (0.0, 10.0), [2.2, 2.8], [0.0, 0.0], &config);
    assert_eq!(result, Err(SimError::MaxStepsExceeded { t: 0.25 }));

    // The budget is per call: a longer run can be made in several.
    let mut stepper = Stepper::new(&dp, 0.0, [2.2, 2.8], [0.0, 0.0], config).unwrap();
    let mut traj = Trajectory::new(stepper.state());
    assert!(traj.extend(&mut stepper, 10.0).is_err());
    assert_eq!(traj.len_segments(), 25);
    traj.extend(&mut stepper, 0.5).unwrap();
    assert_eq!((traj.len_segments(), traj.t_end()), (50, 0.5));
}

#[test]
fn a_span_that_runs_backward_is_refused_and_an_empty_one_is_empty() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let run = |span| simulate(&osc, span, [1.0, 0.0], [0.0, 0.0], &SimConfig::default());
    assert!(matches!(
        run((1.0, 0.0)),
        Err(SimError::InvalidConfig { .. })
    ));
    assert!(matches!(
        run((0.0, f64::NAN)),
        Err(SimError::InvalidConfig { .. })
    ));
    let empty = run((2.0, 2.0)).unwrap();
    assert_eq!(
        (empty.len_segments(), empty.t_start(), empty.t_end()),
        (0, 2.0, 2.0)
    );
    assert_eq!(empty.sample(123.0).q, [1.0, 0.0]);
    assert_eq!(empty.end(), empty.start());
    assert!(
        empty
            .crossings(|s| s.q[0], CrossingDirection::Either)
            .is_empty()
    );
    assert_eq!(empty.action(&osc), 0.0);
}

#[test]
fn crossings_find_events_in_the_right_direction() {
    let pendulum = Pendulum { l: 1.0, g: 9.81 };
    let period = pendulum.period(1.0);
    let traj = simulate(&pendulum, (0.0, 5.25 * period), [1.0], [0.0], &tight()).unwrap();

    // Released at +1: through the bottom going down at T/4, up at 3T/4, …
    let falling = traj.crossings(|s| s.q[0], CrossingDirection::Falling);
    let rising = traj.crossings(|s| s.q[0], CrossingDirection::Rising);
    let either = traj.crossings(|s| s.q[0], CrossingDirection::Either);
    assert_eq!((falling.len(), rising.len(), either.len()), (6, 5, 11));
    let mut worst = 0.0_f64;
    for (k, s) in falling.iter().enumerate() {
        assert!(s.dq[0] < 0.0);
        worst = worst.max((s.t - (0.25 + k as f64) * period).abs() / period);
        worst = worst.max(s.q[0].abs());
    }
    for (k, s) in rising.iter().enumerate() {
        assert!(s.dq[0] > 0.0);
        worst = worst.max((s.t - (0.75 + k as f64) * period).abs() / period);
    }
    check(
        "pendulum: event times vs quarter periods (tight)",
        worst,
        2e-11,
    );
    // In order of time, and "either" is the two interleaved.
    assert!(either.windows(2).all(|w| w[0].t < w[1].t));

    // Turning points are where the velocity changes sign: at 0, T/2, T, …
    // The one at t = 0 starts exactly on zero and is not a crossing.
    let turns = traj.crossings(|s| s.dq[0], CrossingDirection::Either);
    assert_eq!(turns.len(), 10);
    assert!(turns.iter().all(|s| (s.q[0].abs() - 1.0).abs() < 1e-9));
}

#[test]
fn two_crossings_inside_one_step_are_both_found() {
    // A free particle is integrated in very few, very long steps. An
    // indicator that dips below zero and comes back within one of them has
    // the same sign at both ends of the step.
    struct Free;
    impl Lagrangian<1> for Free {
        fn lagrangian<S: Scalar>(&self, _q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5
        }
    }
    let config = SimConfig::default().with_initial_step(10.0);
    let traj = simulate(&Free, (0.0, 10.0), [0.0], [1.0], &config).unwrap();
    assert_eq!(traj.len_segments(), 1, "one step covers the whole run");

    // Negative for x in (3.5, 4.5), positive at both ends. The default
    // looks at x = 0, 2, 4, 6, 8, 10, and x = 4 is inside the dip.
    let dip = |s: &State<1>| (s.q[0] - 3.5) * (s.q[0] - 4.5);
    let knots_only = traj.crossings_with(dip, CrossingDirection::Either, 0);
    assert!(knots_only.is_empty(), "the ends alone cannot see it");
    let found = traj.crossings(dip, CrossingDirection::Either);
    assert_eq!(found.len(), 2);
    assert!((found[0].t - 3.5).abs() < 1e-12 && (found[1].t - 4.5).abs() < 1e-12);
    assert_eq!(traj.crossings(dip, CrossingDirection::Falling).len(), 1);
    assert_eq!(traj.crossings(dip, CrossingDirection::Rising).len(), 1);

    // A dip that fits between two of those points is missed, as documented,
    // and found by looking more finely.
    let narrow = |s: &State<1>| (s.q[0] - 4.4) * (s.q[0] - 4.6);
    assert!(traj.crossings(narrow, CrossingDirection::Either).is_empty());
    assert_eq!(
        traj.crossings_with(narrow, CrossingDirection::Either, 99)
            .len(),
        2
    );
    // Many more points than needed find the same two, once each.
    assert_eq!(
        traj.crossings_with(narrow, CrossingDirection::Either, 100_000)
            .len(),
        2
    );
}

#[test]
fn a_poincare_section_of_a_regular_motion_lies_on_a_curve() {
    // Small oscillations of the double pendulum are quasi-periodic, so the
    // section θ₁ = 0, θ̇₁ > 0 is a closed curve in (θ₂, p₂) and not a cloud:
    // every point has the energy of the motion.
    let dp = DoublePendulum::default();
    let traj = simulate(&dp, (0.0, 200.0), [0.3, -0.2], [0.0, 0.0], &tight()).unwrap();
    let section = traj.crossings(|s| s.q[0], CrossingDirection::Rising);
    assert!(section.len() > 50, "{} points", section.len());
    let e0 = traj.energy(&dp, 0.0).total;
    let scale = traj.energy_drift_at_knots(&dp).scale;
    let worst = section
        .iter()
        .map(|s| {
            assert!(s.q[0].abs() < 1e-10 && s.dq[0] > 0.0);
            (energy(&dp, s.q, s.dq, s.t).total - e0).abs() / scale
        })
        .fold(0.0, f64::max);
    check(
        "energy on a Poincaré section, small oscillations (tight)",
        worst,
        2e-10,
    );
}

#[test]
fn phase_points_and_energy_can_be_read_at_any_time() {
    let dp = DoublePendulum::default();
    let traj = chaotic(&tight());
    let s = traj.sample(4.2);
    assert_eq!(traj.momentum(&dp, 4.2), momentum(&dp, s.q, s.dq, s.t));
    assert_eq!(
        traj.phase_point(&dp, 4.2),
        (s.q, momentum(&dp, s.q, s.dq, s.t))
    );
    assert_eq!(traj.energy(&dp, 4.2), energy(&dp, s.q, s.dq, s.t));
}
