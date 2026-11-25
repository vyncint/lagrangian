//! The stepper on its own: everything here runs without an allocator.

mod common;

use common::{CoupledOscillators, Pendulum, check, rel, tight};
use lagrangian::prelude::*;
use lagrangian::systems::{DoublePendulum, KeplerOrbit};

#[test]
fn energy_is_conserved_one_segment_at_a_time() {
    // The whole simulation loop on a target with no heap: step, look at the
    // segment, forget it.
    let dp = DoublePendulum::default();
    let e0 = energy(&dp, [2.2, 2.8], [0.0, 0.0], 0.0);
    let mut stepper = Stepper::new(&dp, 0.0, [2.2, 2.8], [0.0, 0.0], tight()).unwrap();
    let (mut knots, mut mid, mut kinetic) = (0.0_f64, 0.0_f64, 0.0_f64);
    while stepper.state().t < 20.0 {
        let segment = stepper.step_bounded(20.0).unwrap();
        let end = segment.end();
        let e = energy(&dp, end.q, end.dq, end.t);
        knots = knots.max((e.total - e0.total).abs());
        kinetic = kinetic.max(e.kinetic);
        let half = segment.eval(0.5);
        mid = mid.max((energy(&dp, half.q, half.dq, half.t).total - e0.total).abs());
    }
    let scale = e0.total.abs().max(kinetic);
    assert_eq!(stepper.state().t, 20.0);
    check(
        "stepper: double pendulum energy at knots (tight)",
        knots / scale,
        1e-11,
    );
    check(
        "stepper: double pendulum energy mid-step (tight)",
        mid / scale,
        1e-10,
    );
}

#[test]
fn a_bounded_step_lands_exactly_and_never_overshoots() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    for config in [
        SimConfig::default(),
        SimConfig::variational4(0.3),
        SimConfig::variational2(0.07),
    ] {
        let mut stepper = Stepper::new(&osc, 0.0, [1.0, 0.0], [0.0, 0.0], config).unwrap();
        for target in [0.5, 0.5000001, 2.0, 7.25] {
            let mut previous = stepper.state().t;
            while stepper.state().t < target {
                let segment = stepper.step_bounded(target).unwrap();
                assert_eq!(segment.t_start(), previous);
                assert!(segment.t_end() > segment.t_start());
                assert!(segment.t_end() <= target);
                previous = segment.t_end();
            }
            assert_eq!(stepper.state().t, target);
        }
        // A target that is not ahead is a mistake, not a no-op.
        let now = stepper.state().t;
        for bad in [now, now - 1.0, f64::NAN] {
            assert!(matches!(
                stepper.step_bounded(bad),
                Err(SimError::InvalidConfig { .. })
            ));
        }
    }
}

#[test]
fn fixed_steps_do_not_let_the_clock_drift() {
    // 0.1 is not representable; adding it ten thousand times is off by
    // thousands of ulps. Counting steps from an anchor is not.
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let mut stepper = Stepper::new(
        &osc,
        0.0,
        [1.0, 0.0],
        [0.0, 0.0],
        SimConfig::variational2(0.1),
    )
    .unwrap();
    let mut summed = 0.0;
    for n in 1..=10_000 {
        stepper.step().unwrap();
        summed += 0.1;
        assert_eq!(stepper.state().t, n as f64 * 0.1);
    }
    assert_ne!(summed, 1000.0, "the naive sum has drifted");
    assert_eq!(stepper.state().t, 1000.0);
}

#[test]
fn the_solvers_agree_with_each_other() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let run = |config| {
        let mut s = Stepper::new(&osc, 0.0, [1.0, 0.0], [0.0, 0.0], config).unwrap();
        while s.state().t < 5.0 {
            s.step_bounded(5.0).unwrap();
        }
        s.state()
    };
    let exact = osc.exact(5.0);
    let adaptive = run(tight());
    let fourth = run(SimConfig::variational4(0.01));
    let second = run(SimConfig::variational2(0.01));
    let err = |s: State<2>| (s.q[0] - exact[0]).abs().max((s.q[1] - exact[1]).abs());
    check(
        "dopri5 (tight) vs normal modes at t = 5",
        err(adaptive),
        3e-12,
    );
    check(
        "variational4 dt=0.01 vs normal modes at t = 5",
        err(fourth),
        1e-7,
    );
    check(
        "variational2 dt=0.01 vs normal modes at t = 5",
        err(second),
        6e-4,
    );
}

#[test]
fn the_variational_solvers_have_their_stated_orders() {
    // Halving the step divides the error by 2^order.
    let pend = Pendulum { l: 1.0, g: 9.81 };
    let reference = {
        let mut s = Stepper::new(&pend, 0.0, [1.0], [0.0], tight()).unwrap();
        while s.state().t < 2.0 {
            s.step_bounded(2.0).unwrap();
        }
        s.state().q[0]
    };
    let error = |config: SimConfig| {
        let mut s = Stepper::new(&pend, 0.0, [1.0], [0.0], config).unwrap();
        while s.state().t < 2.0 {
            s.step_bounded(2.0).unwrap();
        }
        (s.state().q[0] - reference).abs()
    };
    let second = error(SimConfig::variational2(0.02)) / error(SimConfig::variational2(0.01));
    let fourth = error(SimConfig::variational4(0.04)) / error(SimConfig::variational4(0.02));
    println!("measured  convergence ratio, variational2 (expect 4): {second:.3}");
    println!("measured  convergence ratio, variational4 (expect 16): {fourth:.3}");
    assert!((3.6..4.4).contains(&second), "second order: ratio {second}");
    assert!(
        (13.0..19.0).contains(&fourth),
        "fourth order: ratio {fourth}"
    );
}

#[test]
fn a_cyclic_momentum_is_conserved_to_the_last_bit() {
    // The discrete Noether theorem, in the integrator's own variables: the
    // momentum of a coordinate the Lagrangian does not depend on is never
    // touched. Recomputing it from the knot's velocity adds only the
    // rounding of that one recovery, which does not accumulate.
    let orbit = KeplerOrbit::default();
    let (q0, dq0) = ([1.0, 0.0], [0.0, 1.2]);
    let p0 = momentum(&orbit, q0, dq0, 0.0)[1];
    let mut stepper = Stepper::new(&orbit, 0.0, q0, dq0, SimConfig::variational4(0.02)).unwrap();
    let mut worst = 0.0_f64;
    for _ in 0..10_000 {
        stepper.step().unwrap();
        let s = stepper.state();
        worst = worst.max(rel(momentum(&orbit, s.q, s.dq, s.t)[1], p0));
    }
    check(
        "kepler p_theta drift, variational4, 10^4 steps",
        worst,
        5e-15,
    );
}

#[test]
fn set_state_restarts_from_the_new_state_and_keeps_the_old_one_on_error() {
    let orbit = KeplerOrbit::default();
    let mut stepper =
        Stepper::new(&orbit, 0.0, [1.0, 0.0], [0.0, 1.2], SimConfig::default()).unwrap();
    stepper.step().unwrap();
    let t = stepper.state().t;

    stepper.set_state([2.0, 1.0], [0.1, 0.5]).unwrap();
    let s = stepper.state();
    assert_eq!((s.t, s.q, s.dq), (t, [2.0, 1.0], [0.1, 0.5]));
    assert_eq!(s.ddq, accelerations(&orbit, s.q, s.dq, s.t).unwrap());
    let segment = stepper.step().unwrap();
    assert_eq!(segment.start(), s);

    // r = 0 is the centre of force: no acceleration exists there.
    let good = stepper.state();
    assert!(stepper.set_state([0.0, 0.0], [0.0, 0.0]).is_err());
    assert_eq!(stepper.state(), good);
    stepper.step().unwrap();
}

#[test]
fn configurations_that_cannot_run_are_refused() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let start = |config| Stepper::new(&osc, 0.0, [1.0, 0.0], [0.0, 0.0], config).map(|_| ());
    let invalid = |config| matches!(start(config), Err(SimError::InvalidConfig { .. }));
    assert!(start(SimConfig::default()).is_ok());
    assert!(invalid(SimConfig::default().with_tolerances(0.0, 0.0)));
    assert!(invalid(SimConfig::default().with_tolerances(-1e-9, 1e-11)));
    assert!(invalid(
        SimConfig::default().with_tolerances(f64::NAN, 1e-11)
    ));
    assert!(invalid(SimConfig::default().with_max_step(0.0)));
    assert!(invalid(
        SimConfig::default().with_min_step(1.0).with_max_step(0.5)
    ));
    assert!(invalid(SimConfig::default().with_initial_step(-0.1)));
    assert!(invalid(SimConfig::variational4(0.0)));
    assert!(invalid(SimConfig::variational2(-0.01)));
    assert!(invalid(SimConfig::variational4(f64::INFINITY)));
    // Only an absolute tolerance, or only a relative one, is fine.
    assert!(start(SimConfig::default().with_tolerances(0.0, 1e-10)).is_ok());
    assert!(start(SimConfig::default().with_tolerances(1e-10, 0.0)).is_ok());
}

#[test]
fn a_state_that_is_not_finite_is_refused_at_the_door() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    for (q, dq, t) in [
        ([f64::NAN, 0.0], [0.0, 0.0], 0.0),
        ([1.0, 0.0], [f64::INFINITY, 0.0], 0.0),
        ([1.0, 0.0], [0.0, 0.0], f64::NAN),
    ] {
        assert!(matches!(
            Stepper::new(&osc, t, q, dq, SimConfig::default()),
            Err(SimError::NonFiniteState { .. })
        ));
    }
}

#[test]
fn a_solution_that_blows_up_ends_in_an_error_not_a_hang() {
    /// q̈ = q³ reaches infinity in finite time.
    struct Explosive;
    impl Lagrangian<1> for Explosive {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5 + q[0].powi(4) * 0.25
        }
    }
    for config in [SimConfig::default(), SimConfig::variational4(0.01)] {
        let mut stepper = Stepper::new(Explosive, 0.0, [1.0], [1.0], config).unwrap();
        let mut steps = 0;
        let error = loop {
            match stepper.step() {
                Ok(_) => steps += 1,
                Err(e) => break e,
            }
            assert!(steps < 100_000, "the blow-up was never noticed");
        };
        println!(
            "measured  blow-up at t = {:.4} after {steps} steps: {error}",
            stepper.state().t
        );
        assert!(matches!(
            error,
            SimError::StepSizeUnderflow { .. }
                | SimError::NonFiniteState { .. }
                | SimError::NewtonDidNotConverge { .. }
                | SimError::SingularMassMatrix { .. }
        ));
        // The stepper is left at its last good knot.
        assert!(stepper.state().is_finite());
        // The exact blow-up time from q(0) = q̇(0) = 1 is 1.31…: the solver
        // must have got close to it, not given up early.
        assert!(stepper.state().t > 1.2 && stepper.state().t < 1.4);
    }
}

#[test]
fn the_minimum_step_is_respected() {
    /// The same blow-up, with a floor on the step.
    struct Explosive;
    impl Lagrangian<1> for Explosive {
        fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
            dq[0].powi(2) * 0.5 + q[0].powi(4) * 0.25
        }
    }
    let config = SimConfig::default().with_min_step(1e-4);
    let mut stepper = Stepper::new(Explosive, 0.0, [1.0], [1.0], config).unwrap();
    let error = loop {
        match stepper.step() {
            Ok(segment) => assert!(segment.duration() > 1e-4),
            Err(e) => break e,
        }
    };
    assert!(
        matches!(error, SimError::StepSizeUnderflow { .. }),
        "{error}"
    );
}

#[test]
fn the_system_can_be_owned_or_borrowed() {
    let osc = CoupledOscillators { k: 1.0, kc: 0.5 };
    let mut borrowed =
        Stepper::new(&osc, 0.0, [1.0, 0.0], [0.0, 0.0], SimConfig::default()).unwrap();
    let mut owned = Stepper::new(
        CoupledOscillators { k: 1.0, kc: 0.5 },
        0.0,
        [1.0, 0.0],
        [0.0, 0.0],
        SimConfig::default(),
    )
    .unwrap();
    assert_eq!(borrowed.step().unwrap(), owned.step().unwrap());
    // Changing a parameter of an owned system takes effect at the knot.
    owned.system_mut().kc = 5.0;
    let s = owned.state();
    let segment = owned.step().unwrap();
    assert_eq!(
        segment.start().ddq,
        accelerations(owned.system(), s.q, s.dq, s.t).unwrap()
    );
    assert_eq!(owned.into_system().kc, 5.0);
}
