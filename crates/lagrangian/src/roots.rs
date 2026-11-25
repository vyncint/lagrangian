//! Bracketed root finding.

use crate::math;

/// A root of `f` in `[a, b]` by Brent's method, given `fa = f(a)` and
/// `fb = f(b)` of opposite sign.
///
/// Combines bisection, the secant rule and inverse quadratic interpolation:
/// it converges as fast as the smooth methods when they behave and never
/// slower than bisection when they do not. The bracket is kept throughout,
/// so the result is always between `a` and `b`.
pub(crate) fn brent(
    mut f: impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    tol: f64,
) -> f64 {
    let (mut a, mut b, mut c) = (a, b, b);
    let (mut fa, mut fb, mut fc) = (fa, fb, fb);
    let mut d = b - a;
    let mut e = d;

    // Bisection alone halves the bracket each time; this is far more than
    // it needs to exhaust an f64.
    for _ in 0..200 {
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if math::abs(fc) < math::abs(fb) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol1 = 2.0 * f64::EPSILON * math::abs(b) + 0.5 * tol;
        let xm = 0.5 * (c - b);
        if math::abs(xm) <= tol1 || fb == 0.0 {
            return b;
        }
        if math::abs(e) >= tol1 && math::abs(fa) > math::abs(fb) {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = math::abs(p);
            let bound = (3.0 * xm * q - math::abs(tol1 * q)).min(math::abs(e * q));
            if 2.0 * p < bound {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        a = b;
        fa = fb;
        b += if math::abs(d) > tol1 {
            d
        } else if xm >= 0.0 {
            tol1
        } else {
            -tol1
        };
        fb = f(b);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_roots_to_the_requested_tolerance() {
        let f = |x: f64| x * x * x - 2.0 * x - 5.0;
        let root = brent(f, 2.0, 3.0, f(2.0), f(3.0), 1e-15);
        assert!((root - 2.094_551_481_542_326_5).abs() < 1e-14);

        let g = |x: f64| x.cos() - x;
        let root = brent(g, 0.0, 1.0, g(0.0), g(1.0), 1e-15);
        assert!((root - 0.739_085_133_215_160_6).abs() < 1e-14);
    }

    #[test]
    fn stays_in_the_bracket_on_a_hard_function() {
        // Flat almost everywhere: the smooth steps overshoot and bisection
        // has to take over.
        let f = |x: f64| (x - 0.123).powi(9);
        let root = brent(f, -1.0, 1.0, f(-1.0), f(1.0), 1e-15);
        assert!((-1.0..=1.0).contains(&root));
        assert!(f(root).abs() < 1e-30);
    }

    #[test]
    fn an_endpoint_that_is_a_root_is_returned() {
        let f = |x: f64| x - 1.0;
        assert_eq!(brent(f, 0.0, 1.0, -1.0, 0.0, 1e-15), 1.0);
    }
}
