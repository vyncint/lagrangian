//! A Lagrangian mechanics engine: write `L = T - V` in plain generic Rust and
//! get exact equations of motion by forward-mode automatic differentiation.
//!
//! **Pre-release.** This is the project skeleton; the engine lands next.

#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![forbid(unsafe_code)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!(
    "lagrangian needs a source of float math: enable the default `std` feature, \
     or `libm` for targets without the standard library \
     (`default-features = false, features = [\"libm\"]`)"
);
