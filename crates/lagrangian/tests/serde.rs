//! Trajectories as data.
#![cfg(all(feature = "serde", feature = "alloc"))]

mod common;

use lagrangian::prelude::*;
use lagrangian::systems::DoublePendulum;

fn sample() -> Trajectory<2> {
    simulate(
        &DoublePendulum::default(),
        (0.0, 2.0),
        [2.2, 2.8],
        [0.0, 0.0],
        &SimConfig::default(),
    )
    .unwrap()
}

#[test]
fn a_trajectory_survives_json_unchanged() {
    let traj = sample();
    let json = serde_json::to_string(&traj).unwrap();
    let back: Trajectory<2> = serde_json::from_str(&json).unwrap();
    // Every coefficient, bit for bit: floats are written in shortest
    // round-trip form.
    assert_eq!(back, traj);
    for k in 0..=200 {
        let t = 0.01 * k as f64;
        assert_eq!(back.sample(t), traj.sample(t));
    }
    // And through a reader, where field names arrive as owned strings.
    let back: Trajectory<2> = serde_json::from_reader(json.as_bytes()).unwrap();
    assert_eq!(back, traj);
}

#[test]
fn the_json_is_self_describing() {
    let traj = sample();
    let value: serde_json::Value = serde_json::to_value(&traj).unwrap();
    let initial = &value["initial"];
    assert_eq!(initial["t"], 0.0);
    assert_eq!(initial["q"], serde_json::json!([2.2, 2.8]));
    assert_eq!(initial["dq"], serde_json::json!([0.0, 0.0]));
    assert_eq!(initial["ddq"].as_array().unwrap().len(), 2);
    let segments = value["segments"].as_array().unwrap();
    assert_eq!(segments.len(), traj.len_segments());
    let first = &segments[0];
    assert_eq!(first["t0"], 0.0);
    assert!(first["t1"].as_f64().unwrap() > 0.0);
    // One row of six coefficients per coordinate.
    let c = first["c"].as_array().unwrap();
    assert_eq!(c.len(), 2);
    assert!(c.iter().all(|row| row.as_array().unwrap().len() == 6));
    // The constant coefficient is the starting position.
    assert_eq!(c[0][0], 2.2);
}

#[test]
fn states_and_segments_round_trip_on_their_own() {
    let traj = sample();
    let state = traj.sample(1.234);
    let back: State<2> = serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
    assert_eq!(back, state);

    let segment = traj.segments()[3];
    let back: Segment<2> = serde_json::from_str(&serde_json::to_string(&segment).unwrap()).unwrap();
    assert_eq!(back, segment);

    // The tuple form a compact format would use is accepted too.
    let back: State<2> = serde_json::from_str("[1.5, [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]").unwrap();
    assert_eq!(
        back,
        State {
            t: 1.5,
            q: [1.0, 2.0],
            dq: [3.0, 4.0],
            ddq: [5.0, 6.0]
        }
    );
    // Fields in any order.
    let back: State<1> =
        serde_json::from_str(r#"{"ddq":[3.0],"q":[1.0],"t":0.5,"dq":[2.0]}"#).unwrap();
    assert_eq!(
        back,
        State {
            t: 0.5,
            q: [1.0],
            dq: [2.0],
            ddq: [3.0]
        }
    );
}

#[test]
fn data_that_would_break_sampling_is_refused() {
    let refuse = |json: &str, what: &str| {
        let result: Result<Trajectory<1>, _> = serde_json::from_str(json);
        assert!(result.is_err(), "accepted {what}");
    };
    let initial = r#"{"t":0.0,"q":[0.0],"dq":[0.0],"ddq":[0.0]}"#;
    let row = "[[0.0,0.0,0.0,0.0,0.0,0.0]]";
    let segment = |t0: &str, t1: &str| format!(r#"{{"t0":{t0},"t1":{t1},"c":{row}}}"#);
    let with = |segments: &str| format!(r#"{{"initial":{initial},"segments":[{segments}]}}"#);

    // The well-formed version of what follows is accepted.
    let good = with(&format!(
        "{},{}",
        segment("0.0", "1.0"),
        segment("1.0", "2.5")
    ));
    let traj: Trajectory<1> = serde_json::from_str(&good).unwrap();
    assert_eq!((traj.len_segments(), traj.t_end()), (2, 2.5));

    // These two start at the initial time, so nothing but the direction of
    // the segment is wrong with them.
    refuse(
        &with(&segment("0.0", "-0.5")),
        "a segment that runs backward",
    );
    refuse(&with(&segment("0.0", "0.0")), "a segment of no duration");
    refuse(
        &with(&segment("0.5", "1.0")),
        "a segment that does not start at the initial time",
    );
    refuse(
        &with(&format!(
            "{},{}",
            segment("0.0", "1.0"),
            segment("1.5", "2.0")
        )),
        "a gap between segments",
    );
    refuse(
        &with(&format!(
            "{},{}",
            segment("0.0", "1.0"),
            segment("0.5", "2.0")
        )),
        "overlapping segments",
    );
    refuse(
        r#"{"initial":{"t":0.0,"q":[0.0],"dq":[0.0]},"segments":[]}"#,
        "a missing field",
    );
    refuse(
        r#"{"initial":{"t":0.0,"q":[0.0,1.0],"dq":[0.0],"ddq":[0.0]},"segments":[]}"#,
        "a coordinate array of the wrong length",
    );
    refuse(
        r#"{"initial":{"t":0.0,"t":1.0,"q":[0.0],"dq":[0.0],"ddq":[0.0]},"segments":[]}"#,
        "a duplicate field",
    );
    refuse(
        r#"{"initial":{"t":0.0,"q":[0.0],"dq":[0.0],"ddq":[0.0],"extra":1},"segments":[]}"#,
        "an unknown field",
    );
    let short_row = r#"{"t0":0.0,"t1":1.0,"c":[[0.0,0.0,0.0]]}"#;
    refuse(&with(short_row), "a row of three coefficients");

    // A segment on its own is held to the same rule: sampling divides by
    // its duration.
    for (t0, t1) in [("1.0", "0.5"), ("1.0", "1.0")] {
        let alone: Result<Segment<1>, _> = serde_json::from_str(&segment(t0, t1));
        assert!(alone.is_err(), "accepted a segment from {t0} to {t1}");
    }
    let alone: Segment<1> = serde_json::from_str(&segment("1.0", "1.5")).unwrap();
    assert_eq!(alone.duration(), 0.5);

    // A trajectory with the wrong number of coordinates for the type.
    let two: Result<Trajectory<2>, _> = serde_json::from_str(&good);
    assert!(two.is_err());
}

#[test]
fn data_that_would_overflow_inside_a_segment_is_refused() {
    // 0.1.0 accepted both and sampled them to NaN (#8).
    let refuse = |json: &str, what: &str| {
        let result: Result<Trajectory<1>, _> = serde_json::from_str(json);
        let error = result.expect_err(what).to_string();
        assert!(error.contains("segment"), "{what}: {error}");
    };
    let from = |t0: &str, t1: &str, row: &str| {
        format!(
            r#"{{"initial":{{"t":{t0},"q":[0.0],"dq":[0.0],"ddq":[0.0]}},"segments":[{{"t0":{t0},"t1":{t1},"c":[{row}]}}]}}"#
        )
    };
    let zeros = "[0.0,0.0,0.0,0.0,0.0,0.0]";
    refuse(&from("-1e308", "1e308", zeros), "a duration that overflows");
    refuse(
        &from("0.0", "1e10", "[0.0,0.0,0.0,0.0,0.0,1e300]"),
        "a position that overflows inside the segment",
    );
    refuse(
        &from("0.0", "10.0", "[0.0,1e308,0.0,0.0,0.0,0.0]"),
        "a velocity that carries the position past the largest float",
    );
    refuse(
        &from("0.0", "1e100", "[0.0,0.0,0.0,1e100,0.0,0.0]"),
        "an acceleration that overflows",
    );
    // Large but representable everywhere inside is fine.
    let big: Trajectory<1> =
        serde_json::from_str(&from("0.0", "1.0", "[1e300,1e300,0.0,0.0,0.0,0.0]")).unwrap();
    assert!(big.sample(0.5).is_finite());

    let alone: Result<Segment<1>, _> =
        serde_json::from_str(r#"{"t0":-1e308,"t1":1e308,"c":[[0.0,0.0,0.0,0.0,0.0,0.0]]}"#);
    assert!(alone.is_err());
}

#[test]
fn a_trajectory_starts_where_its_first_segment_does() {
    // An initial state that disagrees with the first segment: 0.1.0's
    // sample(0) reported the one and sample(1e-12) the other (#8). The
    // first segment wins, and the file still round-trips unchanged.
    let json = r#"{"initial":{"t":0.0,"q":[5.0],"dq":[5.0],"ddq":[5.0]},
                   "segments":[{"t0":0.0,"t1":1.0,"c":[[0.0,1.0,0.0,0.0,0.0,0.0]]}]}"#;
    let traj: Trajectory<1> = serde_json::from_str(json).unwrap();
    let first = traj.segments()[0].start();
    assert_eq!(first.q, [0.0]);
    assert_eq!(traj.start(), first);
    assert_eq!(traj.sample(0.0), first);
    assert_eq!(traj.sample(-1.0), first);
    assert_eq!(traj.knots().next(), Some(first));
    assert!((traj.sample(1e-12).q[0] - 1e-12).abs() < 1e-24);
    let back: Trajectory<1> = serde_json::from_str(&serde_json::to_string(&traj).unwrap()).unwrap();
    assert_eq!(back, traj);
}

/// Replaces the `k`-th number met in a depth-first walk; `false` if there
/// are not that many.
fn replace_nth(value: &mut serde_json::Value, k: &mut usize, with: f64) -> bool {
    use serde_json::Value;
    match value {
        Value::Number(_) => {
            if *k == 0 {
                *value = serde_json::json!(with);
                return true;
            }
            *k -= 1;
            false
        }
        Value::Array(items) => items.iter_mut().any(|x| replace_nth(x, k, with)),
        Value::Object(fields) => fields.values_mut().any(|x| replace_nth(x, k, with)),
        _ => false,
    }
}

#[test]
fn no_single_number_can_make_a_valid_file_sample_to_nan() {
    // Every number of a real trajectory replaced in turn by values far out
    // of range: whatever is still accepted is finite everywhere inside. In
    // 0.1.0 about 0.3% of randomly mutated files were accepted and sampled
    // to inf or NaN (#8).
    let traj = simulate(
        &DoublePendulum::default(),
        (0.0, 0.2),
        [2.2, 2.8],
        [0.0, 0.0],
        &SimConfig::default(),
    )
    .unwrap();
    let original = serde_json::to_value(&traj).unwrap();
    let (mut tried, mut accepted) = (0, 0);
    for k in 0.. {
        let mut done = false;
        for with in [
            1e308, -1e308, 1e300, 1e154, -1e154, 1e-300, 5e-324, 0.0, 3.0,
        ] {
            let mut value = original.clone();
            if !replace_nth(&mut value, &mut k.clone(), with) {
                done = true;
                break;
            }
            tried += 1;
            let Ok(read) = serde_json::from_value::<Trajectory<2>>(value) else {
                continue;
            };
            accepted += 1;
            for segment in read.segments() {
                for tau in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    assert!(segment.eval(tau).is_finite(), "number {k} set to {with:e}");
                }
            }
            let (t0, span) = (read.t_start(), read.duration());
            for i in 0..=40 {
                let s = read.sample(t0 + span * f64::from(i) / 40.0);
                assert!(s.is_finite(), "number {k} set to {with:e}");
            }
        }
        if done {
            break;
        }
    }
    // Most replacements land in a coefficient and are harmless.
    assert!(
        tried > 1000 && accepted > tried / 2,
        "{accepted} of {tried} accepted"
    );
}
