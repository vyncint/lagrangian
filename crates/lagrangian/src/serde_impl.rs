//! `Serialize` and `Deserialize`, written by hand.
//!
//! serde implements its traits for arrays of thirty-two fixed lengths and
//! not for `[T; N]` with a const-generic `N`, so a derive cannot be used on
//! any type here, and the crate does not take a helper dependency to work
//! around it. Each type is a struct with named fields in self-describing
//! formats (JSON) and a plain tuple in the others; arrays are fixed-length
//! sequences.
//!
//! Nothing is lost in a round trip as long as the format keeps every bit of
//! an `f64`. `serde_json` writes shortest round-trip floats but reads them
//! back exactly only with its `float_roundtrip` feature; without it a
//! coefficient can come back one ulp off.
//!
//! Deserialization checks what the rest of the crate relies on without
//! checking: a segment runs forward in time, and the segments of a
//! trajectory are contiguous. A file that breaks either is refused rather
//! than sampled.

use core::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeStruct, SerializeTuple};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::segment::{Segment, State};

/// `[f64; K]` as a sequence of exactly `K` numbers.
struct Row<const K: usize>([f64; K]);

impl<const K: usize> Serialize for Row<K> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(K)?;
        for x in &self.0 {
            tuple.serialize_element(x)?;
        }
        tuple.end()
    }
}

impl<'de, const K: usize> Deserialize<'de> for Row<K> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct RowVisitor<const K: usize>;

        impl<'de, const K: usize> Visitor<'de> for RowVisitor<K> {
            type Value = Row<K>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "an array of {K} numbers")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Row<K>, A::Error> {
                let mut row = [0.0; K];
                for (i, slot) in row.iter_mut().enumerate() {
                    *slot = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                }
                Ok(Row(row))
            }
        }

        deserializer.deserialize_tuple(K, RowVisitor::<K>)
    }
}

/// `[[f64; 6]; N]` as a sequence of exactly `N` rows.
struct Coefficients<const N: usize>([[f64; 6]; N]);

impl<const N: usize> Serialize for Coefficients<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tuple = serializer.serialize_tuple(N)?;
        for row in &self.0 {
            tuple.serialize_element(&Row(*row))?;
        }
        tuple.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for Coefficients<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct CoefficientsVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for CoefficientsVisitor<N> {
            type Value = Coefficients<N>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "an array of {N} rows of 6 numbers")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Coefficients<N>, A::Error> {
                let mut rows = [[0.0; 6]; N];
                for (i, slot) in rows.iter_mut().enumerate() {
                    let row: Row<6> = seq
                        .next_element()?
                        .ok_or_else(|| de::Error::invalid_length(i, &self))?;
                    *slot = row.0;
                }
                Ok(Coefficients(rows))
            }
        }

        deserializer.deserialize_tuple(N, CoefficientsVisitor::<N>)
    }
}

/// Reads a field name and returns its position in a list of known names.
struct FieldIndex(&'static [&'static str]);

impl<'de> DeserializeSeed<'de> for FieldIndex {
    type Value = usize;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<usize, D::Error> {
        deserializer.deserialize_identifier(self)
    }
}

impl Visitor<'_> for FieldIndex {
    type Value = usize;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a field name")
    }

    fn visit_str<E: de::Error>(self, name: &str) -> Result<usize, E> {
        self.0
            .iter()
            .position(|known| *known == name)
            .ok_or_else(|| de::Error::unknown_field(name, self.0))
    }

    fn visit_bytes<E: de::Error>(self, name: &[u8]) -> Result<usize, E> {
        self.0
            .iter()
            .position(|known| known.as_bytes() == name)
            .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Bytes(name), &self))
    }

    fn visit_u64<E: de::Error>(self, index: u64) -> Result<usize, E> {
        usize::try_from(index)
            .ok()
            .filter(|i| *i < self.0.len())
            .ok_or_else(|| de::Error::invalid_value(de::Unexpected::Unsigned(index), &self))
    }
}

/// Stores a field's value, refusing a second one.
fn set<T, E: de::Error>(slot: &mut Option<T>, value: T, name: &'static str) -> Result<(), E> {
    if slot.is_some() {
        return Err(de::Error::duplicate_field(name));
    }
    *slot = Some(value);
    Ok(())
}

fn required<T, E: de::Error>(slot: Option<T>, name: &'static str) -> Result<T, E> {
    slot.ok_or_else(|| de::Error::missing_field(name))
}

/// The next element of a tuple-shaped struct, or a length error.
fn element<'de, T, A, V>(seq: &mut A, index: usize, visitor: &V) -> Result<T, A::Error>
where
    T: Deserialize<'de>,
    A: SeqAccess<'de>,
    V: Visitor<'de>,
{
    seq.next_element()?
        .ok_or_else(|| de::Error::invalid_length(index, visitor))
}

// ---------------------------------------------------------------------------
// State<N>
// ---------------------------------------------------------------------------

const STATE_FIELDS: &[&str] = &["t", "q", "dq", "ddq"];

impl<const N: usize> Serialize for State<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("State", STATE_FIELDS.len())?;
        st.serialize_field("t", &self.t)?;
        st.serialize_field("q", &Row(self.q))?;
        st.serialize_field("dq", &Row(self.dq))?;
        st.serialize_field("ddq", &Row(self.ddq))?;
        st.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for State<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StateVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for StateVisitor<N> {
            type Value = State<N>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a state of {N} coordinates")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<State<N>, A::Error> {
                let t = element(&mut seq, 0, &self)?;
                let q: Row<N> = element(&mut seq, 1, &self)?;
                let dq: Row<N> = element(&mut seq, 2, &self)?;
                let ddq: Row<N> = element(&mut seq, 3, &self)?;
                Ok(State {
                    t,
                    q: q.0,
                    dq: dq.0,
                    ddq: ddq.0,
                })
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<State<N>, A::Error> {
                let (mut t, mut q, mut dq, mut ddq) = (None, None, None, None);
                while let Some(field) = map.next_key_seed(FieldIndex(STATE_FIELDS))? {
                    match field {
                        0 => set(&mut t, map.next_value()?, "t")?,
                        1 => set(&mut q, map.next_value::<Row<N>>()?.0, "q")?,
                        2 => set(&mut dq, map.next_value::<Row<N>>()?.0, "dq")?,
                        _ => set(&mut ddq, map.next_value::<Row<N>>()?.0, "ddq")?,
                    }
                }
                Ok(State {
                    t: required(t, "t")?,
                    q: required(q, "q")?,
                    dq: required(dq, "dq")?,
                    ddq: required(ddq, "ddq")?,
                })
            }
        }

        deserializer.deserialize_struct("State", STATE_FIELDS, StateVisitor::<N>)
    }
}

// ---------------------------------------------------------------------------
// Segment<N>
// ---------------------------------------------------------------------------

const SEGMENT_FIELDS: &[&str] = &["t0", "t1", "c"];

impl<const N: usize> Serialize for Segment<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut st = serializer.serialize_struct("Segment", SEGMENT_FIELDS.len())?;
        st.serialize_field("t0", &self.t0)?;
        st.serialize_field("t1", &self.t1)?;
        st.serialize_field("c", &Coefficients(self.c))?;
        st.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for Segment<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct SegmentVisitor<const N: usize>;

        impl<const N: usize> SegmentVisitor<N> {
            /// Sampling divides by the duration, so it must be positive.
            fn build<E: de::Error>(t0: f64, t1: f64, c: [[f64; 6]; N]) -> Result<Segment<N>, E> {
                if t1.partial_cmp(&t0) != Some(core::cmp::Ordering::Greater)
                    || !t0.is_finite()
                    || !t1.is_finite()
                {
                    return Err(de::Error::custom(
                        "a segment must run forward in time: t1 must be after t0, and both finite",
                    ));
                }
                Ok(Segment { t0, t1, c })
            }
        }

        impl<'de, const N: usize> Visitor<'de> for SegmentVisitor<N> {
            type Value = Segment<N>;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "a trajectory segment of {N} coordinates")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Segment<N>, A::Error> {
                let t0 = element(&mut seq, 0, &self)?;
                let t1 = element(&mut seq, 1, &self)?;
                let c: Coefficients<N> = element(&mut seq, 2, &self)?;
                Self::build(t0, t1, c.0)
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Segment<N>, A::Error> {
                let (mut t0, mut t1, mut c) = (None, None, None);
                while let Some(field) = map.next_key_seed(FieldIndex(SEGMENT_FIELDS))? {
                    match field {
                        0 => set(&mut t0, map.next_value()?, "t0")?,
                        1 => set(&mut t1, map.next_value()?, "t1")?,
                        _ => set(&mut c, map.next_value::<Coefficients<N>>()?.0, "c")?,
                    }
                }
                Self::build(required(t0, "t0")?, required(t1, "t1")?, required(c, "c")?)
            }
        }

        deserializer.deserialize_struct("Segment", SEGMENT_FIELDS, SegmentVisitor::<N>)
    }
}

// ---------------------------------------------------------------------------
// Trajectory<N>
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
mod trajectory {
    use alloc::vec::Vec;
    use core::marker::PhantomData;

    use super::*;
    use crate::trajectory::Trajectory;

    const TRAJECTORY_FIELDS: &[&str] = &["initial", "segments"];

    impl<const N: usize> Serialize for Trajectory<N> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let mut st = serializer.serialize_struct("Trajectory", TRAJECTORY_FIELDS.len())?;
            st.serialize_field("initial", &self.initial)?;
            st.serialize_field("segments", &self.segments[..])?;
            st.end()
        }
    }

    struct TrajectoryVisitor<const N: usize>(PhantomData<[(); N]>);

    impl<const N: usize> TrajectoryVisitor<N> {
        /// Sampling finds a segment by binary search over times, which is
        /// only right when each segment starts where the one before ended.
        fn build<E: de::Error>(
            initial: State<N>,
            segments: Vec<Segment<N>>,
        ) -> Result<Trajectory<N>, E> {
            let mut end = initial.t;
            for segment in &segments {
                if segment.t0 != end {
                    return Err(de::Error::custom(
                        "the segments of a trajectory must be contiguous, starting at the initial time",
                    ));
                }
                end = segment.t1;
            }
            Ok(Trajectory { initial, segments })
        }
    }

    impl<'de, const N: usize> Visitor<'de> for TrajectoryVisitor<N> {
        type Value = Trajectory<N>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "a trajectory of {N} coordinates")
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Trajectory<N>, A::Error> {
            let initial = element(&mut seq, 0, &self)?;
            let segments = element(&mut seq, 1, &self)?;
            Self::build(initial, segments)
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Trajectory<N>, A::Error> {
            let (mut initial, mut segments) = (None, None);
            while let Some(field) = map.next_key_seed(FieldIndex(TRAJECTORY_FIELDS))? {
                match field {
                    0 => set(&mut initial, map.next_value()?, "initial")?,
                    _ => set(&mut segments, map.next_value()?, "segments")?,
                }
            }
            Self::build(
                required(initial, "initial")?,
                required(segments, "segments")?,
            )
        }
    }

    impl<'de, const N: usize> Deserialize<'de> for Trajectory<N> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_struct(
                "Trajectory",
                TRAJECTORY_FIELDS,
                TrajectoryVisitor::<N>(PhantomData),
            )
        }
    }
}
