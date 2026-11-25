//! The number type a [`Lagrangian`](crate::Lagrangian) is written against.

use core::fmt::Debug;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math;

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// A real number that may carry derivatives along with its value.
///
/// A Lagrangian is written once, generically over `S: Scalar`. The engine
/// then calls it with plain `f64` to get a value and with the dual-number
/// types of [`crate::ad`] to get exact derivatives. You never name those
/// types: bound on `Scalar` and use ordinary arithmetic.
///
/// # Writing constants
///
/// Inside generic code a scalar can be combined with an `f64` on its **right**:
/// `x * 0.5`, `x + self.offset`, `v_sq * (0.5 * self.m)`. With the `f64` on
/// the left (`0.5 * x`) the compiler has no bound to find, so either flip the
/// operands or lift the constant with `S::from_f64(0.5)`.
///
/// # Branching
///
/// Comparison (`<`, `==`, …) looks at the value only and ignores the
/// derivative parts, so a branch such as `if x > S::ZERO` picks the same arm
/// for every scalar type. [`value`](Scalar::value) returns that value.
///
/// # Derivatives at singular points
///
/// Mechanical systems start at rest and pass through the origin, so three
/// functions are defined there on purpose:
///
/// - [`powi`](Scalar::powi) is repeated multiplication, never
///   `n * x.powi(n - 1)`, so `x.powi(2)` has a clean derivative at `x == 0`.
/// - [`hypot`](Scalar::hypot), [`atan2`](Scalar::atan2) and
///   [`abs`](Scalar::abs) return zero derivatives at the origin instead of
///   `0.0 / 0.0`.
/// - [`sqrt`](Scalar::sqrt) is **not** guarded: its derivative at zero is
///   infinite, and the result is `NaN` or infinite. Write a norm as
///   `x.hypot(y)`, and keep a speed squared rather than rooting it.
///
/// # Sealed
///
/// This trait cannot be implemented outside the crate, which lets methods be
/// added to it without a breaking release.
pub trait Scalar:
    sealed::Sealed
    + Copy
    + Debug
    + PartialEq
    + PartialOrd
    + From<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// The constant `0`.
    const ZERO: Self;
    /// The constant `1`.
    const ONE: Self;

    /// Lifts a constant: its derivatives are zero.
    fn from_f64(x: f64) -> Self;
    /// The value, with every derivative part dropped.
    fn value(self) -> f64;

    /// `1 / self`.
    fn recip(self) -> Self;
    /// Square root. Not guarded at zero; see the trait documentation.
    fn sqrt(self) -> Self;
    /// Sine, in radians.
    fn sin(self) -> Self;
    /// Cosine, in radians.
    fn cos(self) -> Self;
    /// Sine and cosine together.
    fn sin_cos(self) -> (Self, Self) {
        (self.sin(), self.cos())
    }
    /// Tangent, in radians.
    fn tan(self) -> Self;
    /// Arcsine.
    fn asin(self) -> Self;
    /// Arccosine.
    fn acos(self) -> Self;
    /// Arctangent.
    fn atan(self) -> Self;
    /// The angle of the point `(x, self)`: `self` is the `y` coordinate, as
    /// in [`f64::atan2`]. Derivatives are zero at the origin.
    fn atan2(self, x: Self) -> Self;
    /// Hyperbolic sine.
    fn sinh(self) -> Self;
    /// Hyperbolic cosine.
    fn cosh(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// `e^self`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Absolute value. The derivative at zero is zero.
    fn abs(self) -> Self;
    /// `sqrt(self² + other²)` without overflow. Derivatives are zero at the
    /// origin.
    fn hypot(self, other: Self) -> Self;

    /// Integer power by repeated multiplication. Safe at `self == 0` for
    /// `n >= 0`; a negative `n` is the reciprocal of the positive power.
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }

    /// Real power. An exponent that is a whole number is routed to
    /// [`powi`](Scalar::powi), so `x.powf(2.0)` is as safe at zero as
    /// `x.powi(2)`. Otherwise the derivative is `p * x^(p - 1)`, which is
    /// not finite at `x == 0` for `p < 1`.
    fn powf(self, p: f64) -> Self;
}

/// `x^n` using only `*` and `recip`, so that every derivative order comes
/// out of the product rule and none out of `n * (n - 1) * x^(n - 2)`.
///
/// The closed-form derivative is the bug this exists to avoid: at rest
/// (`x == 0`) with `n == 1` it evaluates `0 * 0^-1 = 0 * inf = NaN`.
pub(crate) fn powi_by_squaring<S: Scalar>(x: S, n: i32) -> S {
    let mut exp = n.unsigned_abs();
    let mut base = x;
    let mut acc = S::ONE;
    let mut started = false;
    while exp > 0 {
        if exp & 1 == 1 {
            // The first factor is taken as is: `ONE * base` is the same
            // number, but the multiply would turn a `-0.0` into `0.0`.
            acc = if started { acc * base } else { base };
            started = true;
        }
        exp >>= 1;
        if exp > 0 {
            base = base * base;
        }
    }
    if n < 0 { acc.recip() } else { acc }
}

/// `p` as an `i32` when it is exactly a whole number in range.
pub(crate) fn whole_exponent(p: f64) -> Option<i32> {
    if (-1024.0..=1024.0).contains(&p) {
        let n = p as i32;
        if n as f64 == p {
            return Some(n);
        }
    }
    None
}

/// `1`, `-1` or `0`: the derivative of `abs`, with the kink given slope zero.
pub(crate) fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

impl sealed::Sealed for f64 {}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn value(self) -> f64 {
        self
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }
    #[inline]
    fn sqrt(self) -> Self {
        math::sqrt(self)
    }
    #[inline]
    fn sin(self) -> Self {
        math::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        math::cos(self)
    }
    #[inline]
    fn tan(self) -> Self {
        math::tan(self)
    }
    #[inline]
    fn asin(self) -> Self {
        math::asin(self)
    }
    #[inline]
    fn acos(self) -> Self {
        math::acos(self)
    }
    #[inline]
    fn atan(self) -> Self {
        math::atan(self)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        math::atan2(self, x)
    }
    #[inline]
    fn sinh(self) -> Self {
        math::sinh(self)
    }
    #[inline]
    fn cosh(self) -> Self {
        math::cosh(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        math::tanh(self)
    }
    #[inline]
    fn exp(self) -> Self {
        math::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        math::ln(self)
    }
    #[inline]
    fn abs(self) -> Self {
        math::abs(self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        math::hypot(self, other)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        match whole_exponent(p) {
            Some(n) => powi_by_squaring(self, n),
            None => math::powf(self, p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powi_matches_repeated_multiplication() {
        for &x in &[0.0, 1.0, -1.5, 2.25, 1e-3] {
            assert_eq!(Scalar::powi(x, 0), 1.0);
            assert_eq!(Scalar::powi(x, 1), x);
            assert_eq!(Scalar::powi(x, 2), x * x);
            assert_eq!(Scalar::powi(x, 3), x * x * x);
            assert_eq!(Scalar::powi(x, 5), (x * x) * (x * x) * x);
        }
        assert_eq!(Scalar::powi(2.0, -3), 0.125);
        assert_eq!(Scalar::powi(2.0, 10), 1024.0);
        assert_eq!(Scalar::powi(-2.0, 3), -8.0);
    }

    #[test]
    fn powi_survives_the_extreme_exponent() {
        // i32::MIN has no positive counterpart; unsigned_abs is what keeps
        // this from overflowing.
        assert_eq!(Scalar::powi(1.0, i32::MIN), 1.0);
        assert_eq!(Scalar::powi(2.0, i32::MIN), 0.0);
    }

    #[test]
    fn whole_exponents_are_recognised() {
        assert_eq!(whole_exponent(2.0), Some(2));
        assert_eq!(whole_exponent(-3.0), Some(-3));
        assert_eq!(whole_exponent(0.0), Some(0));
        assert_eq!(whole_exponent(0.5), None);
        assert_eq!(whole_exponent(1e9), None);
        assert_eq!(whole_exponent(f64::NAN), None);
        assert_eq!(Scalar::powf(3.0, 2.0), 9.0);
        assert!((Scalar::powf(4.0, 0.5) - 2.0).abs() < 1e-15);
    }

    #[test]
    fn sign_gives_the_kink_slope_zero() {
        assert_eq!(sign(3.0), 1.0);
        assert_eq!(sign(-3.0), -1.0);
        assert_eq!(sign(0.0), 0.0);
        assert_eq!(sign(-0.0), 0.0);
    }
}
