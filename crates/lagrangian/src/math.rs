//! Float math from `std` when it is there, from `libm` when it is not.
//!
//! Everything in the crate that needs a transcendental function goes through
//! here, so the choice is made in one place and the `no-std` gates have one
//! file to break.

macro_rules! unary {
    ($($name:ident => $std:ident, $libm:ident;)*) => {
        $(
            #[inline]
            pub(crate) fn $name(x: f64) -> f64 {
                #[cfg(feature = "std")]
                {
                    x.$std()
                }
                #[cfg(all(not(feature = "std"), feature = "libm"))]
                {
                    libm::$libm(x)
                }
                // Neither: lib.rs has already refused to compile, and says
                // why. Returning the argument keeps that the only error.
                #[cfg(not(any(feature = "std", feature = "libm")))]
                {
                    x
                }
            }
        )*
    };
}

macro_rules! binary {
    ($($name:ident => $std:ident, $libm:ident;)*) => {
        $(
            #[inline]
            pub(crate) fn $name(x: f64, y: f64) -> f64 {
                #[cfg(feature = "std")]
                {
                    x.$std(y)
                }
                #[cfg(all(not(feature = "std"), feature = "libm"))]
                {
                    libm::$libm(x, y)
                }
                #[cfg(not(any(feature = "std", feature = "libm")))]
                {
                    x + y
                }
            }
        )*
    };
}

unary! {
    sin => sin, sin;
    cos => cos, cos;
    tan => tan, tan;
    asin => asin, asin;
    acos => acos, acos;
    atan => atan, atan;
    sinh => sinh, sinh;
    cosh => cosh, cosh;
    tanh => tanh, tanh;
    exp => exp, exp;
    ln => ln, log;
    sqrt => sqrt, sqrt;
    abs => abs, fabs;
}

binary! {
    atan2 => atan2, atan2;
    hypot => hypot, hypot;
    powf => powf, pow;
}

/// `x` reduced into `[0, period)`. `f64::rem_euclid` lives in `std`.
#[cfg(feature = "alloc")]
#[inline]
pub(crate) fn rem_euclid(x: f64, period: f64) -> f64 {
    let r = x % period;
    if r < 0.0 { r + period } else { r }
}
