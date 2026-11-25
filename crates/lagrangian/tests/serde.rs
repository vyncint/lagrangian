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
