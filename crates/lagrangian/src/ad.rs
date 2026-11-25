//! Forward-mode automatic differentiation on the stack.
//!
//! **This module is exempt from semver until 1.0.** It is public so the
//! types can be inspected and tested, but user code should bound on
//! [`Scalar`] and never name them: that is what keeps the
//! engine free to change how it differentiates.
//!
//! Three types, all `Copy` and none of them allocating:
//!
//! - [`Dual<T>`]: a value and one directional derivative, generic over what
//!   it wraps. Nesting it is how higher orders are built:
//!   [`HyperDual`] = `Dual<Dual<f64>>` is the scalar reference oracle, and
//!   `Dual<Grad<N>>` is what the engine uses.
//! - [`Grad<N>`]: a value and its gradient with respect to `N` variables.
//! - [`Taylor2`]: a truncated Taylor polynomial `c0 + c1·τ + c2·τ²`, which
//!   turns position, velocity and half the acceleration of every coordinate
//!   into the same three numbers for any point of the mechanism.
//!
//! Comparison (`==`, `<`, …) on all three looks at the value only.

// The product and quotient rules put a `+` inside `Mul` and a `*` inside
// `Div`. That is the calculus, not a typo.
#![allow(clippy::suspicious_arithmetic_impl)]

use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::math;
use crate::scalar::{Scalar, powi_by_squaring, sealed::Sealed, sign, whole_exponent};

/// The assignment operators and the `f64`-on-the-left operators, which are
/// the same few lines for every type here.
macro_rules! derived_ops {
    ([$($gen:tt)*] $ty:ty) => {
        impl<$($gen)*> AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl<$($gen)*> SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl<$($gen)*> MulAssign for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl<$($gen)*> DivAssign for $ty {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
        impl<$($gen)*> AddAssign<f64> for $ty {
            #[inline]
            fn add_assign(&mut self, rhs: f64) {
                *self = *self + rhs;
            }
        }
        impl<$($gen)*> SubAssign<f64> for $ty {
            #[inline]
            fn sub_assign(&mut self, rhs: f64) {
                *self = *self - rhs;
            }
        }
        impl<$($gen)*> MulAssign<f64> for $ty {
            #[inline]
            fn mul_assign(&mut self, rhs: f64) {
                *self = *self * rhs;
            }
        }
        impl<$($gen)*> DivAssign<f64> for $ty {
            #[inline]
            fn div_assign(&mut self, rhs: f64) {
                *self = *self / rhs;
            }
        }
        impl<$($gen)*> Add<$ty> for f64 {
            type Output = $ty;
            #[inline]
            fn add(self, rhs: $ty) -> $ty {
                rhs + self
            }
        }
        impl<$($gen)*> Sub<$ty> for f64 {
            type Output = $ty;
            #[inline]
            fn sub(self, rhs: $ty) -> $ty {
                -rhs + self
            }
        }
        impl<$($gen)*> Mul<$ty> for f64 {
            type Output = $ty;
            #[inline]
            fn mul(self, rhs: $ty) -> $ty {
                rhs * self
            }
        }
        impl<$($gen)*> Div<$ty> for f64 {
            type Output = $ty;
            #[inline]
            fn div(self, rhs: $ty) -> $ty {
                rhs.recip() * self
            }
        }
        impl<$($gen)*> From<f64> for $ty {
            #[inline]
            fn from(x: f64) -> Self {
                <Self as Scalar>::from_f64(x)
            }
        }
        impl<$($gen)*> PartialEq for $ty {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.value() == other.value()
            }
        }
        impl<$($gen)*> PartialOrd for $ty {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.value().partial_cmp(&other.value())
            }
        }
        impl<$($gen)*> Sealed for $ty {}
    };
}

// ---------------------------------------------------------------------------
// Dual<T>
// ---------------------------------------------------------------------------

/// A value and its derivative along one direction: `val + eps·ε` with
/// `ε² = 0`.
///
/// Generic over the number it wraps, so derivatives of derivatives are a
/// matter of nesting.
#[derive(Debug, Clone, Copy)]
pub struct Dual<T> {
    /// The value.
    pub val: T,
    /// The derivative along the seeded direction.
    pub eps: T,
}

/// A scalar second-order dual number: value, two first derivatives and one
/// mixed second derivative. The independent oracle the engine's extraction
/// is checked against.
pub type HyperDual = Dual<Dual<f64>>;

impl<T: Scalar> Dual<T> {
    /// A value with the given derivative.
    #[inline]
    pub const fn new(val: T, eps: T) -> Self {
        Self { val, eps }
    }

    /// A constant: derivative zero.
    #[inline]
    pub fn constant(val: T) -> Self {
        Self { val, eps: T::ZERO }
    }

    /// The variable being differentiated against: derivative one.
    #[inline]
    pub fn variable(val: T) -> Self {
        Self { val, eps: T::ONE }
    }

    /// `f(self)` given `f(val)` and `f'(val)`.
    #[inline]
    fn chain(self, f: T, df: T) -> Self {
        Self {
            val: f,
            eps: self.eps * df,
        }
    }
}

impl<T: Scalar> Add for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            val: self.val + rhs.val,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<T: Scalar> Sub for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            val: self.val - rhs.val,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<T: Scalar> Mul for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            val: self.val * rhs.val,
            eps: self.val * rhs.eps + self.eps * rhs.val,
        }
    }
}

impl<T: Scalar> Div for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = rhs.val.recip();
        let val = self.val * inv;
        Self {
            val,
            eps: (self.eps - val * rhs.eps) * inv,
        }
    }
}

impl<T: Scalar> Neg for Dual<T> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            eps: -self.eps,
        }
    }
}

impl<T: Scalar> Add<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self {
            val: self.val + rhs,
            eps: self.eps,
        }
    }
}

impl<T: Scalar> Sub<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self {
            val: self.val - rhs,
            eps: self.eps,
        }
    }
}

impl<T: Scalar> Mul<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            val: self.val * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl<T: Scalar> Div<f64> for Dual<T> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            val: self.val / rhs,
            eps: self.eps / rhs,
        }
    }
}

derived_ops!([T: Scalar] Dual<T>);

impl<T: Scalar> Scalar for Dual<T> {
    const ZERO: Self = Self {
        val: T::ZERO,
        eps: T::ZERO,
    };
    const ONE: Self = Self {
        val: T::ONE,
        eps: T::ZERO,
    };

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(T::from_f64(x))
    }
    #[inline]
    fn value(self) -> f64 {
        self.val.value()
    }
    #[inline]
    fn recip(self) -> Self {
        let r = self.val.recip();
        self.chain(r, -(r * r))
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.val.sqrt();
        self.chain(s, (s * 2.0).recip())
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain(s, c)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = self.val.sin_cos();
        self.chain(c, -s)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = self.val.sin_cos();
        (self.chain(s, c), self.chain(c, -s))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = self.val.tan();
        self.chain(t, t * t + 1.0)
    }
    #[inline]
    fn asin(self) -> Self {
        let v = self.val;
        self.chain(v.asin(), (T::ONE - v * v).sqrt().recip())
    }
    #[inline]
    fn acos(self) -> Self {
        let v = self.val;
        self.chain(v.acos(), -(T::ONE - v * v).sqrt().recip())
    }
    #[inline]
    fn atan(self) -> Self {
        let v = self.val;
        self.chain(v.atan(), (v * v + 1.0).recip())
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        let val = self.val.atan2(x.val);
        // (x·dy − y·dx) / r², with x and y scaled by r = hypot(x, y) before
        // anything is squared: x² + y² underflows to zero below
        // |(x, y)| ≈ 1e-154, where the derivative is enormous, and 0.1.0
        // returned zero there (#16).
        let r = self.val.hypot(x.val);
        if r.value() == 0.0 {
            return Self::constant(val);
        }
        let (xs, ys) = (x.val / r, self.val / r);
        Self {
            val,
            eps: (xs * self.eps - ys * x.eps) / r,
        }
    }
    #[inline]
    fn sinh(self) -> Self {
        self.chain(self.val.sinh(), self.val.cosh())
    }
    #[inline]
    fn cosh(self) -> Self {
        self.chain(self.val.cosh(), self.val.sinh())
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = self.val.tanh();
        self.chain(t, T::ONE - t * t)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.val.exp();
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(self.val.ln(), self.val.recip())
    }
    #[inline]
    fn abs(self) -> Self {
        let s = sign(self.val.value());
        Self {
            val: self.val.abs(),
            eps: self.eps * s,
        }
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let h = self.val.hypot(other.val);
        if h.value() == 0.0 {
            return Self::constant(h);
        }
        // (a·da + b·db) / h, with a and b divided by h first: a product of a
        // tiny value and a tiny derivative underflows, as in `atan2` (#16).
        Self {
            val: h,
            eps: (self.val / h) * self.eps + (other.val / h) * other.eps,
        }
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        match whole_exponent(p) {
            Some(n) => powi_by_squaring(self, n),
            None => self.chain(self.val.powf(p), self.val.powf(p - 1.0) * p),
        }
    }
}

// ---------------------------------------------------------------------------
// Grad<N>
// ---------------------------------------------------------------------------

/// A value and its gradient with respect to `N` variables.
#[derive(Debug, Clone, Copy)]
pub struct Grad<const N: usize> {
    /// The value.
    pub val: f64,
    /// The partial derivatives.
    pub d: [f64; N],
}

impl<const N: usize> Grad<N> {
    /// A constant: gradient zero.
    #[inline]
    pub const fn constant(val: f64) -> Self {
        Self { val, d: [0.0; N] }
    }

    /// The `index`-th variable: gradient is the `index`-th unit vector.
    ///
    /// # Panics
    ///
    /// If `index >= N`.
    #[inline]
    pub fn variable(val: f64, index: usize) -> Self {
        let mut d = [0.0; N];
        d[index] = 1.0;
        Self { val, d }
    }

    /// `f(self)` given `f(val)` and `f'(val)`.
    #[inline]
    fn chain(self, f: f64, df: f64) -> Self {
        Self {
            val: f,
            d: self.d.map(|x| x * df),
        }
    }

    #[inline]
    fn zip(self, rhs: Self, val: f64, f: impl Fn(f64, f64) -> f64) -> Self {
        Self {
            val,
            d: core::array::from_fn(|i| f(self.d[i], rhs.d[i])),
        }
    }
}

impl<const N: usize> Add for Grad<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.zip(rhs, self.val + rhs.val, |a, b| a + b)
    }
}

impl<const N: usize> Sub for Grad<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.zip(rhs, self.val - rhs.val, |a, b| a - b)
    }
}

impl<const N: usize> Mul for Grad<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.val, rhs.val);
        self.zip(rhs, a * b, |da, db| a * db + da * b)
    }
}

impl<const N: usize> Div for Grad<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let inv = 1.0 / rhs.val;
        let val = self.val * inv;
        self.zip(rhs, val, |da, db| (da - val * db) * inv)
    }
}

impl<const N: usize> Neg for Grad<N> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            val: -self.val,
            d: self.d.map(|x| -x),
        }
    }
}

impl<const N: usize> Add<f64> for Grad<N> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self {
            val: self.val + rhs,
            d: self.d,
        }
    }
}

impl<const N: usize> Sub<f64> for Grad<N> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self {
            val: self.val - rhs,
            d: self.d,
        }
    }
}

impl<const N: usize> Mul<f64> for Grad<N> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            val: self.val * rhs,
            d: self.d.map(|x| x * rhs),
        }
    }
}

impl<const N: usize> Div<f64> for Grad<N> {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self {
            val: self.val / rhs,
            d: self.d.map(|x| x / rhs),
        }
    }
}

derived_ops!([const N: usize] Grad<N>);

impl<const N: usize> Scalar for Grad<N> {
    const ZERO: Self = Self::constant(0.0);
    const ONE: Self = Self::constant(1.0);

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(x)
    }
    #[inline]
    fn value(self) -> f64 {
        self.val
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.val;
        self.chain(r, -(r * r))
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = math::sqrt(self.val);
        self.chain(s, 1.0 / (2.0 * s))
    }
    #[inline]
    fn sin(self) -> Self {
        self.chain(math::sin(self.val), math::cos(self.val))
    }
    #[inline]
    fn cos(self) -> Self {
        self.chain(math::cos(self.val), -math::sin(self.val))
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = (math::sin(self.val), math::cos(self.val));
        (self.chain(s, c), self.chain(c, -s))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = math::tan(self.val);
        self.chain(t, t * t + 1.0)
    }
    #[inline]
    fn asin(self) -> Self {
        let v = self.val;
        self.chain(math::asin(v), 1.0 / math::sqrt(1.0 - v * v))
    }
    #[inline]
    fn acos(self) -> Self {
        let v = self.val;
        self.chain(math::acos(v), -1.0 / math::sqrt(1.0 - v * v))
    }
    #[inline]
    fn atan(self) -> Self {
        let v = self.val;
        self.chain(math::atan(v), 1.0 / (v * v + 1.0))
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        let val = math::atan2(self.val, x.val);
        // Scaled before squaring, as for `Dual` (#16).
        let r = math::hypot(self.val, x.val);
        if r == 0.0 {
            return Self::constant(val);
        }
        let (ys, xs) = (self.val / r, x.val / r);
        self.zip(x, val, |dy, dx| (xs * dy - ys * dx) / r)
    }
    #[inline]
    fn sinh(self) -> Self {
        self.chain(math::sinh(self.val), math::cosh(self.val))
    }
    #[inline]
    fn cosh(self) -> Self {
        self.chain(math::cosh(self.val), math::sinh(self.val))
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = math::tanh(self.val);
        self.chain(t, 1.0 - t * t)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = math::exp(self.val);
        self.chain(e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        self.chain(math::ln(self.val), 1.0 / self.val)
    }
    #[inline]
    fn abs(self) -> Self {
        self.chain(math::abs(self.val), sign(self.val))
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let h = math::hypot(self.val, other.val);
        if h == 0.0 {
            return Self::constant(0.0);
        }
        // Scaled before multiplying, as for `Dual`.
        let (ua, ub) = (self.val / h, other.val / h);
        self.zip(other, h, |da, db| ua * da + ub * db)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        match whole_exponent(p) {
            Some(n) => powi_by_squaring(self, n),
            None => self.chain(math::powf(self.val, p), p * math::powf(self.val, p - 1.0)),
        }
    }
}

// ---------------------------------------------------------------------------
// Taylor2
// ---------------------------------------------------------------------------

/// A Taylor polynomial truncated after the quadratic term:
/// `c0 + c1·τ + c2·τ²  (mod τ³)`.
///
/// Seeding a coordinate with `(q, q̇, q̈/2)` and pushing it through a
/// position map gives the position, velocity and half the acceleration of
/// the mapped point in one evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Taylor2 {
    /// The value.
    pub c0: f64,
    /// The first derivative.
    pub c1: f64,
    /// Half the second derivative.
    pub c2: f64,
}

impl Taylor2 {
    /// A polynomial from its three coefficients.
    #[inline]
    pub const fn new(c0: f64, c1: f64, c2: f64) -> Self {
        Self { c0, c1, c2 }
    }

    /// A constant.
    #[inline]
    pub const fn constant(c0: f64) -> Self {
        Self {
            c0,
            c1: 0.0,
            c2: 0.0,
        }
    }

    /// `f(self)` given `f(c0)`, `f'(c0)` and `f''(c0)`.
    #[inline]
    fn compose(self, f: f64, df: f64, ddf: f64) -> Self {
        Self {
            c0: f,
            c1: df * self.c1,
            c2: df * self.c2 + 0.5 * ddf * self.c1 * self.c1,
        }
    }
}

impl Add for Taylor2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.c0 + rhs.c0, self.c1 + rhs.c1, self.c2 + rhs.c2)
    }
}

impl Sub for Taylor2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.c0 - rhs.c0, self.c1 - rhs.c1, self.c2 - rhs.c2)
    }
}

impl Mul for Taylor2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.c0 * rhs.c0,
            self.c0 * rhs.c1 + self.c1 * rhs.c0,
            self.c0 * rhs.c2 + self.c1 * rhs.c1 + self.c2 * rhs.c0,
        )
    }
}

impl Div for Taylor2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        // Solve (self) = (quotient)·(rhs) coefficient by coefficient.
        let inv = 1.0 / rhs.c0;
        let c0 = self.c0 * inv;
        let c1 = (self.c1 - c0 * rhs.c1) * inv;
        let c2 = (self.c2 - c0 * rhs.c2 - c1 * rhs.c1) * inv;
        Self::new(c0, c1, c2)
    }
}

impl Neg for Taylor2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.c0, -self.c1, -self.c2)
    }
}

impl Add<f64> for Taylor2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f64) -> Self {
        Self::new(self.c0 + rhs, self.c1, self.c2)
    }
}

impl Sub<f64> for Taylor2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f64) -> Self {
        Self::new(self.c0 - rhs, self.c1, self.c2)
    }
}

impl Mul<f64> for Taylor2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.c0 * rhs, self.c1 * rhs, self.c2 * rhs)
    }
}

impl Div<f64> for Taylor2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        Self::new(self.c0 / rhs, self.c1 / rhs, self.c2 / rhs)
    }
}

derived_ops!([] Taylor2);

impl Scalar for Taylor2 {
    const ZERO: Self = Self::constant(0.0);
    const ONE: Self = Self::constant(1.0);

    #[inline]
    fn from_f64(x: f64) -> Self {
        Self::constant(x)
    }
    #[inline]
    fn value(self) -> f64 {
        self.c0
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.c0;
        self.compose(r, -(r * r), 2.0 * r * r * r)
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = math::sqrt(self.c0);
        let ds = 1.0 / (2.0 * s);
        self.compose(s, ds, -ds / (2.0 * self.c0))
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = (math::sin(self.c0), math::cos(self.c0));
        self.compose(s, c, -s)
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = (math::sin(self.c0), math::cos(self.c0));
        self.compose(c, -s, -c)
    }
    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (s, c) = (math::sin(self.c0), math::cos(self.c0));
        (self.compose(s, c, -s), self.compose(c, -s, -c))
    }
    #[inline]
    fn tan(self) -> Self {
        let t = math::tan(self.c0);
        let sec2 = 1.0 + t * t;
        self.compose(t, sec2, 2.0 * t * sec2)
    }
    #[inline]
    fn asin(self) -> Self {
        let x = self.c0;
        let w = 1.0 - x * x;
        let d = 1.0 / math::sqrt(w);
        self.compose(math::asin(x), d, x * d / w)
    }
    #[inline]
    fn acos(self) -> Self {
        let x = self.c0;
        let w = 1.0 - x * x;
        let d = 1.0 / math::sqrt(w);
        self.compose(math::acos(x), -d, -(x * d / w))
    }
    #[inline]
    fn atan(self) -> Self {
        let x = self.c0;
        let d = 1.0 / (1.0 + x * x);
        self.compose(math::atan(x), d, -2.0 * x * d * d)
    }
    #[inline]
    fn atan2(self, x: Self) -> Self {
        let c0 = math::atan2(self.c0, x.c0);
        // atan2 is unchanged when both arguments are divided by the same
        // positive number, and so are the coefficients below. Dividing by
        // r = hypot(x, y) first keeps x² + y² near one instead of letting it
        // underflow (#16).
        let r = math::hypot(self.c0, x.c0);
        if r == 0.0 {
            return Self::constant(c0);
        }
        let (y, x) = (self / r, x / r);
        let d0 = x.c0 * x.c0 + y.c0 * y.c0;
        // θ' = (x·y' − y·x') / (x² + y²), expanded to first order in τ and
        // integrated once.
        let n0 = x.c0 * y.c1 - y.c0 * x.c1;
        let n1 = 2.0 * (x.c0 * y.c2 - y.c0 * x.c2);
        let d1 = 2.0 * (x.c0 * x.c1 + y.c0 * y.c1);
        let t1 = n0 / d0;
        Self::new(c0, t1, 0.5 * (n1 - t1 * d1) / d0)
    }
    #[inline]
    fn sinh(self) -> Self {
        let (s, c) = (math::sinh(self.c0), math::cosh(self.c0));
        self.compose(s, c, s)
    }
    #[inline]
    fn cosh(self) -> Self {
        let (s, c) = (math::sinh(self.c0), math::cosh(self.c0));
        self.compose(c, s, c)
    }
    #[inline]
    fn tanh(self) -> Self {
        let t = math::tanh(self.c0);
        let d = 1.0 - t * t;
        self.compose(t, d, -2.0 * t * d)
    }
    #[inline]
    fn exp(self) -> Self {
        let e = math::exp(self.c0);
        self.compose(e, e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        let r = 1.0 / self.c0;
        self.compose(math::ln(self.c0), r, -(r * r))
    }
    #[inline]
    fn abs(self) -> Self {
        self.compose(math::abs(self.c0), sign(self.c0), 0.0)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        let h = math::hypot(self.c0, other.c0);
        if h == 0.0 {
            return Self::constant(0.0);
        }
        // The square root of a² + b² about h², with (a, b) divided by h
        // before anything is multiplied, as for `Dual`. `a'² + b'² − c1²` is
        // the square of the part of (a', b') across (a, b), and is computed
        // directly instead of as a difference.
        let (ua, ub) = (self.c0 / h, other.c0 / h);
        let c1 = ua * self.c1 + ub * other.c1;
        let w = ua * other.c1 - ub * self.c1;
        Self::new(h, c1, 0.5 * w * (w / h) + ua * self.c2 + ub * other.c2)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        powi_by_squaring(self, n)
    }
    #[inline]
    fn powf(self, p: f64) -> Self {
        match whole_exponent(p) {
            Some(n) => powi_by_squaring(self, n),
            None => {
                let x = self.c0;
                self.compose(
                    math::powf(x, p),
                    p * math::powf(x, p - 1.0),
                    p * (p - 1.0) * math::powf(x, p - 2.0),
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A function exercising every method, written once for every scalar.
    fn everything<S: Scalar>(x: S, y: S) -> S {
        let a = (x * y + x.sin() * y.cos()).exp() / (x * x + y * y + 1.5);
        let b = (x.tan() - y.tanh()) * (x.sinh() * x.sinh() + y.cosh()).ln() + (x * y).abs();
        let c = (x * 0.3).asin() + (y * 0.2).acos() + (x / y).atan() + y.atan2(x);
        let d = x.hypot(y) + (x * x + 2.0).sqrt() + x.powi(3) - y.powi(-2) + (y + 3.0).powf(1.7);
        let e = (x - y).recip() * 0.25 - (-x) + x.powf(2.0);
        a + b + c + d + e
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    const POINTS: [(f64, f64); 4] = [(0.7, -1.3), (-0.4, 0.9), (1.1, 2.2), (-1.9, -0.6)];

    #[test]
    fn dual_matches_central_differences() {
        for &(x, y) in &POINTS {
            let h = 1e-6;
            let fx = everything(Dual::variable(x), Dual::constant(y));
            let fy = everything(Dual::constant(x), Dual::variable(y));
            let nx = (everything(x + h, y) - everything(x - h, y)) / (2.0 * h);
            let ny = (everything(x, y + h) - everything(x, y - h)) / (2.0 * h);
            assert_eq!(fx.val, everything(x, y));
            assert!(close(fx.eps, nx, 1e-7), "d/dx {} vs {}", fx.eps, nx);
            assert!(close(fy.eps, ny, 1e-7), "d/dy {} vs {}", fy.eps, ny);
        }
    }

    #[test]
    fn grad_agrees_with_dual() {
        for &(x, y) in &POINTS {
            let g = everything(Grad::<2>::variable(x, 0), Grad::<2>::variable(y, 1));
            let fx = everything(Dual::variable(x), Dual::constant(y));
            let fy = everything(Dual::constant(x), Dual::variable(y));
            assert!(close(g.val, fx.val, 1e-14));
            assert!(close(g.d[0], fx.eps, 1e-13));
            assert!(close(g.d[1], fy.eps, 1e-13));
        }
    }

    #[test]
    fn hyperdual_gives_the_mixed_second_derivative() {
        for &(x, y) in &POINTS {
            // Outer dual along x, inner along y: eps.eps is d²f/dxdy.
            let xx = Dual::new(Dual::constant(x), Dual::constant(1.0));
            let yy = Dual::new(Dual::variable(y), Dual::constant(0.0));
            let f: HyperDual = everything(xx, yy);
            let h = 1e-5;
            let dy = |x: f64| everything(Dual::constant(x), Dual::variable(y)).eps;
            let mixed = (dy(x + h) - dy(x - h)) / (2.0 * h);
            assert!(
                close(f.eps.eps, mixed, 1e-7),
                "d2/dxdy {} vs {}",
                f.eps.eps,
                mixed
            );
        }
    }

    #[test]
    fn taylor2_matches_the_hyperdual_along_a_path() {
        for &(x, y) in &POINTS {
            // x(τ) = x + 0.3τ − 0.2τ², y(τ) = y − 0.7τ + 0.5τ².
            let t = everything(Taylor2::new(x, 0.3, -0.2), Taylor2::new(y, -0.7, 0.5));
            // The same path as a hyperdual with both levels seeded along τ:
            // the mixed part of a coordinate is its second derivative.
            let hx = Dual::new(Dual::new(x, 0.3), Dual::new(0.3, 2.0 * -0.2));
            let hy = Dual::new(Dual::new(y, -0.7), Dual::new(-0.7, 2.0 * 0.5));
            let f: HyperDual = everything(hx, hy);
            assert!(close(t.c0, f.val.val, 1e-14));
            assert!(
                close(t.c1, f.val.eps, 1e-12),
                "c1 {} vs {}",
                t.c1,
                f.val.eps
            );
            assert!(
                close(t.c2, 0.5 * f.eps.eps, 1e-12),
                "c2 {} vs {}",
                t.c2,
                0.5 * f.eps.eps
            );
        }
    }

    #[test]
    fn powi_is_clean_at_rest() {
        // The closed-form second derivative of x¹ is 1·0·x⁻¹ = 0·inf = NaN
        // at x = 0. Repeated multiplication never forms it.
        for n in 0..6 {
            let x = Dual::new(Dual::variable(0.0), Dual::constant(1.0));
            let f: HyperDual = x.powi(n);
            assert!(f.val.val.is_finite() && f.val.eps.is_finite());
            assert!(f.eps.val.is_finite() && f.eps.eps.is_finite(), "n = {n}");
            let t = Taylor2::new(0.0, 1.0, 0.5).powi(n);
            assert!(t.c0.is_finite() && t.c1.is_finite() && t.c2.is_finite());
        }
        let x = Dual::new(Dual::variable(0.0), Dual::constant(1.0));
        let sq: HyperDual = x.powi(2);
        assert_eq!(
            (sq.val.val, sq.val.eps, sq.eps.val, sq.eps.eps),
            (0.0, 0.0, 0.0, 2.0)
        );
        // powf with a whole exponent takes the same route.
        let sq: HyperDual = x.powf(2.0);
        assert_eq!(sq.eps.eps, 2.0);
        // A negative velocity is fine too: no logarithm is involved.
        let g = Grad::<1>::variable(-3.0, 0).powi(3);
        assert_eq!((g.val, g.d[0]), (-27.0, 27.0));
    }

    #[test]
    fn hypot_atan2_and_abs_are_clean_at_the_origin() {
        let zero = Dual::new(Dual::variable(0.0), Dual::constant(1.0));
        let h: HyperDual = zero.hypot(zero);
        assert_eq!(
            (h.val.val, h.val.eps, h.eps.val, h.eps.eps),
            (0.0, 0.0, 0.0, 0.0)
        );
        let a: HyperDual = zero.atan2(zero);
        assert!(a.val.eps == 0.0 && a.eps.val == 0.0 && a.eps.eps == 0.0);
        let b: HyperDual = zero.abs();
        assert_eq!((b.val.eps, b.eps.val), (0.0, 0.0));

        let g0 = Grad::<2>::variable(0.0, 0);
        let g1 = Grad::<2>::variable(0.0, 1);
        assert_eq!(g0.hypot(g1).d, [0.0, 0.0]);
        assert_eq!(g0.atan2(g1).d, [0.0, 0.0]);

        let t = Taylor2::new(0.0, 1.0, 0.5);
        let h = t.hypot(t);
        assert_eq!((h.c0, h.c1, h.c2), (0.0, 0.0, 0.0));
        let a = t.atan2(t);
        assert_eq!((a.c1, a.c2), (0.0, 0.0));
    }

    #[test]
    fn atan2_and_hypot_keep_their_derivatives_for_tiny_arguments() {
        // 0.1.0 divided by x² + y², which underflows to zero below about
        // 1e-154, and then reported a derivative of zero (#16). `hypot`
        // multiplied each value by its derivative first, which underflows
        // when both are tiny.
        let close = |a: f64, b: f64| (a - b).abs() <= 4.0 * f64::EPSILON * b.abs();

        // ∂/∂y atan2(y, x) = x / (x² + y²) = 5e169 at x = y = 1e-170.
        let g = Grad::<2>::variable(1e-170, 0).atan2(Grad::variable(1e-170, 1));
        assert!(close(g.d[0], 5e169) && close(g.d[1], -5e169), "{:?}", g.d);

        // atan2(1e-170·(1 + s), 1e-170) = atan(1 + s): slope 1/2, and
        // second derivative -1/2 at s = 0.
        let y = Dual::new(1e-170, 1e-170);
        let d = y.atan2(Dual::constant(1e-170));
        assert!(close(d.eps, 0.5), "{}", d.eps);
        let x = Dual::new(Dual::constant(1e-170), Dual::constant(0.0));
        let y = Dual::new(Dual::new(1e-170, 1e-170), Dual::constant(1e-170));
        let h: HyperDual = y.atan2(x);
        assert!(close(h.eps.val, 0.5) && close(h.val.eps, 0.5), "{h:?}");
        assert!(close(h.eps.eps, -0.5), "{}", h.eps.eps);

        let t = Taylor2::new(1e-170, 1e-170, 0.0).atan2(Taylor2::constant(1e-170));
        assert!(close(t.c1, 0.5) && close(t.c2, -0.25), "{t:?}");

        // hypot(1e-170·(1 + s), 1e-170) = 1e-170·√((1 + s)² + 1): slope
        // 1e-170/√2 and second derivative 1e-170/(2√2) at s = 0.
        let h = Dual::new(1e-170, 1e-170).hypot(Dual::constant(1e-170));
        assert!(close(h.eps, 1e-170 / core::f64::consts::SQRT_2), "{h:?}");
        let t = Taylor2::new(1e-170, 1e-170, 0.0).hypot(Taylor2::constant(1e-170));
        assert!(
            close(t.c1, 1e-170 / core::f64::consts::SQRT_2)
                && close(t.c2, 1e-170 / (4.0 * core::f64::consts::SQRT_2)),
            "{t:?}"
        );

        // Nothing changes at ordinary sizes.
        let g = Grad::<2>::variable(0.3, 0).atan2(Grad::variable(-0.4, 1));
        let r2 = 0.3_f64 * 0.3 + 0.4 * 0.4;
        assert!(
            close(g.d[0], -0.4 / r2) && close(g.d[1], -0.3 / r2),
            "{:?}",
            g.d
        );
        let t = Taylor2::new(0.3, 0.7, -0.2).hypot(Taylor2::new(-0.4, 0.1, 0.5));
        // √(a² + b²) with a = 0.3 + 0.7s − 0.2s², b = −0.4 + 0.1s + 0.5s².
        let (c1, c2) = ((0.3 * 0.7 - 0.4 * 0.1) / 0.5, 0.0);
        let w = (0.3 * 0.1 + 0.4 * 0.7) / 0.5;
        let c2 = c2 + 0.5 * w * w / 0.5 + (0.3 * -0.2 - 0.4 * 0.5) / 0.5;
        assert!(
            close(t.c1, c1) && (t.c2 - c2).abs() <= 1e-15,
            "{t:?} vs {c1}, {c2}"
        );
    }

    #[test]
    fn sqrt_at_zero_is_not_masked() {
        // The derivative is infinite, and the result must say so rather
        // than report a finite slope that is not there.
        let g = Grad::<1>::variable(0.0, 0).sqrt();
        assert_eq!(g.val, 0.0);
        assert!(!g.d[0].is_finite());
    }

    #[test]
    fn comparison_looks_at_the_value_only() {
        let a = Dual::new(1.0, 5.0);
        let b = Dual::new(1.0, -5.0);
        assert!(a == b);
        assert!(Dual::new(0.5, 9.0) < a);
        assert!(Grad::<2>::variable(2.0, 0) > Grad::<2>::constant(1.0));
        assert!(Taylor2::new(1.0, 2.0, 3.0) == Taylor2::constant(1.0));
    }

    #[test]
    fn f64_on_the_left_works_for_concrete_types() {
        let x = Dual::variable(2.0);
        let y = 3.0 * x + 1.0 - x;
        assert_eq!((y.val, y.eps), (5.0, 2.0));
        let z = 1.0 / x;
        assert_eq!((z.val, z.eps), (0.5, -0.25));
        let w = 10.0 - Grad::<1>::variable(4.0, 0);
        assert_eq!((w.val, w.d[0]), (6.0, -1.0));
    }
}
