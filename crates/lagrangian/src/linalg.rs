//! Small dense linear solves on the stack.
//!
//! The mass matrix of a mechanical system is symmetric, but it is **not**
//! always positive definite: the "mass matrix" of a geodesic Lagrangian is
//! the metric, and a spacetime metric has one negative eigenvalue. So there
//! is no Cholesky here. Everything is Gaussian elimination with partial
//! pivoting, which does not care about the sign of a pivot.

use crate::math;

/// A pivot at or below this fraction of the matrix's largest entry is
/// treated as zero.
const SINGULAR_RATIO: f64 = 64.0 * f64::EPSILON;

/// An `N × N` matrix factored as `PA = LU` with partial pivoting.
///
/// Factor once, then [`solve`](Lu::solve) for as many right-hand sides as
/// needed.
#[derive(Debug, Clone, Copy)]
pub struct Lu<const N: usize> {
    /// `L` below the diagonal (unit diagonal implied), `U` on and above it.
    lu: [[f64; N]; N],
    /// Row `i` of the factorization is row `perm[i]` of the original.
    perm: [usize; N],
}

impl<const N: usize> Lu<N> {
    /// Factors `a`, or returns `None` if it is singular to working precision
    /// or contains a value that is not finite.
    pub fn factor(a: &[[f64; N]; N]) -> Option<Self> {
        let tol = singular_tolerance(a)?;
        let mut lu = *a;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);

        for k in 0..N {
            // The largest entry in column k, at or below the diagonal.
            let mut pivot_row = k;
            let mut pivot_abs = math::abs(lu[k][k]);
            for (r, row) in lu.iter().enumerate().skip(k + 1) {
                let candidate = math::abs(row[k]);
                if candidate > pivot_abs {
                    pivot_abs = candidate;
                    pivot_row = r;
                }
            }
            // Written so that a NaN pivot fails the test.
            if pivot_abs.partial_cmp(&tol) != Some(core::cmp::Ordering::Greater) {
                return None;
            }
            if pivot_row != k {
                lu.swap(k, pivot_row);
                perm.swap(k, pivot_row);
            }
            let pivot = lu[k][k];
            for r in (k + 1)..N {
                let factor = lu[r][k] / pivot;
                lu[r][k] = factor;
                for c in (k + 1)..N {
                    lu[r][c] -= factor * lu[k][c];
                }
            }
        }
        Some(Self { lu, perm })
    }

    /// Solves `A x = b`.
    pub fn solve(&self, b: &[f64; N]) -> [f64; N] {
        let mut x: [f64; N] = core::array::from_fn(|i| b[self.perm[i]]);
        // L y = P b
        for i in 0..N {
            for j in 0..i {
                x[i] -= self.lu[i][j] * x[j];
            }
        }
        // U x = y
        for i in (0..N).rev() {
            for j in (i + 1)..N {
                x[i] -= self.lu[i][j] * x[j];
            }
            x[i] /= self.lu[i][i];
        }
        x
    }
}

/// Solves `A x = b`, or returns `None` if `A` is singular to working
/// precision or contains a value that is not finite.
///
/// `N = 1, 2, 3` take fixed-size paths with no permutation bookkeeping;
/// every other size goes through [`Lu`]. All of them pivot, so an indefinite
/// matrix is as welcome as a positive definite one.
pub fn solve<const N: usize>(a: &[[f64; N]; N], b: &[f64; N]) -> Option<[f64; N]> {
    match N {
        0 => Some([0.0; N]),
        1 => {
            let tol = singular_tolerance(a)?;
            if !is_above(a[0][0], tol) {
                return None;
            }
            let mut x = [0.0; N];
            x[0] = b[0] / a[0][0];
            Some(x)
        }
        2 => {
            let tol = singular_tolerance(a)?;
            let s = solve2([[a[0][0], a[0][1]], [a[1][0], a[1][1]]], [b[0], b[1]], tol)?;
            let mut x = [0.0; N];
            x[..2].copy_from_slice(&s);
            Some(x)
        }
        3 => {
            let tol = singular_tolerance(a)?;
            let m = [
                [a[0][0], a[0][1], a[0][2]],
                [a[1][0], a[1][1], a[1][2]],
                [a[2][0], a[2][1], a[2][2]],
            ];
            let s = solve3(m, [b[0], b[1], b[2]], tol)?;
            let mut x = [0.0; N];
            x[..3].copy_from_slice(&s);
            Some(x)
        }
        _ => Lu::factor(a).map(|lu| lu.solve(b)),
    }
}

/// `SINGULAR_RATIO` times the largest magnitude in `a`; `None` if any entry
/// is not finite or all of them are zero.
fn singular_tolerance<const N: usize>(a: &[[f64; N]; N]) -> Option<f64> {
    let mut largest = 0.0_f64;
    for row in a {
        for &x in row {
            if !x.is_finite() {
                return None;
            }
            let m = math::abs(x);
            if m > largest {
                largest = m;
            }
        }
    }
    if N > 0 && largest == 0.0 {
        return None;
    }
    Some(SINGULAR_RATIO * largest)
}

#[inline]
fn is_above(pivot: f64, tol: f64) -> bool {
    math::abs(pivot) > tol
}

fn solve2(a: [[f64; 2]; 2], b: [f64; 2], tol: f64) -> Option<[f64; 2]> {
    // Pivot on the larger entry of the first column.
    let (r0, r1, b0, b1) = if math::abs(a[0][0]) >= math::abs(a[1][0]) {
        (a[0], a[1], b[0], b[1])
    } else {
        (a[1], a[0], b[1], b[0])
    };
    if !is_above(r0[0], tol) {
        return None;
    }
    let f = r1[0] / r0[0];
    let u11 = r1[1] - f * r0[1];
    if !is_above(u11, tol) {
        return None;
    }
    let x1 = (b1 - f * b0) / u11;
    let x0 = (b0 - r0[1] * x1) / r0[0];
    Some([x0, x1])
}

fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3], tol: f64) -> Option<[f64; 3]> {
    // Column 0.
    let mut p = 0;
    if math::abs(a[1][0]) > math::abs(a[p][0]) {
        p = 1;
    }
    if math::abs(a[2][0]) > math::abs(a[p][0]) {
        p = 2;
    }
    a.swap(0, p);
    b.swap(0, p);
    if !is_above(a[0][0], tol) {
        return None;
    }
    let f1 = a[1][0] / a[0][0];
    let f2 = a[2][0] / a[0][0];
    a[1][1] -= f1 * a[0][1];
    a[1][2] -= f1 * a[0][2];
    b[1] -= f1 * b[0];
    a[2][1] -= f2 * a[0][1];
    a[2][2] -= f2 * a[0][2];
    b[2] -= f2 * b[0];

    // Column 1.
    if math::abs(a[2][1]) > math::abs(a[1][1]) {
        a.swap(1, 2);
        b.swap(1, 2);
    }
    if !is_above(a[1][1], tol) {
        return None;
    }
    let f = a[2][1] / a[1][1];
    a[2][2] -= f * a[1][2];
    b[2] -= f * b[1];
    if !is_above(a[2][2], tol) {
        return None;
    }

    let x2 = b[2] / a[2][2];
    let x1 = (b[1] - a[1][2] * x2) / a[1][1];
    let x0 = (b[0] - a[0][1] * x1 - a[0][2] * x2) / a[0][0];
    Some([x0, x1, x2])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator: the tests must not depend on a
    /// crate, and must give the same matrices on every machine.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// A symmetric matrix with eigenvalues of both signs and a condition
    /// number of a few tens: `Qᵀ D Q` would need an orthogonal `Q`, so this
    /// takes the cheaper route of a signed diagonal plus a small symmetric
    /// perturbation.
    fn symmetric_indefinite<const N: usize>(rng: &mut Lcg) -> [[f64; N]; N] {
        let mut a = [[0.0; N]; N];
        for i in 0..N {
            let sign = if i % 2 == 0 { -1.0 } else { 1.0 };
            a[i][i] = sign * (1.0 + 0.5 * rng.next().abs()) * (1.0 + i as f64);
            for j in 0..i {
                let off = 0.15 * rng.next();
                a[i][j] = off;
                a[j][i] = off;
            }
        }
        a
    }

    fn residual<const N: usize>(a: &[[f64; N]; N], x: &[f64; N], b: &[f64; N]) -> f64 {
        let mut worst = 0.0_f64;
        let mut scale = 0.0_f64;
        for i in 0..N {
            let ax: f64 = (0..N).map(|j| a[i][j] * x[j]).sum();
            worst = worst.max((ax - b[i]).abs());
            scale = scale.max(b[i].abs());
        }
        worst / scale
    }

    fn fast_path_agrees_with_lu<const N: usize>() {
        let mut rng = Lcg(0x5eed + N as u64);
        for _ in 0..200 {
            let a = symmetric_indefinite::<N>(&mut rng);
            let b: [f64; N] = core::array::from_fn(|_| rng.next() * 3.0);
            let fast = solve(&a, &b).expect("well conditioned");
            let general = Lu::factor(&a).expect("well conditioned").solve(&b);
            for i in 0..N {
                let scale = 1.0_f64.max(fast[i].abs()).max(general[i].abs());
                assert!(
                    (fast[i] - general[i]).abs() <= 1e-12 * scale,
                    "N = {N}: {} vs {}",
                    fast[i],
                    general[i]
                );
            }
            assert!(residual(&a, &fast, &b) <= 1e-12);
        }
    }

    #[test]
    fn fixed_size_paths_agree_with_the_general_factorization() {
        fast_path_agrees_with_lu::<1>();
        fast_path_agrees_with_lu::<2>();
        fast_path_agrees_with_lu::<3>();
    }

    fn general_path_solves<const N: usize>() {
        let mut rng = Lcg(0xfeed + N as u64);
        for _ in 0..100 {
            let a = symmetric_indefinite::<N>(&mut rng);
            let b: [f64; N] = core::array::from_fn(|_| rng.next() * 3.0);
            let x = solve(&a, &b).expect("well conditioned");
            assert!(residual(&a, &x, &b) <= 1e-12, "N = {N}");
        }
    }

    #[test]
    fn the_general_path_solves_indefinite_systems() {
        general_path_solves::<4>();
        general_path_solves::<6>();
        general_path_solves::<10>();
    }

    #[test]
    fn a_lorentzian_metric_is_not_a_problem() {
        // diag(-, +, +, +): Cholesky would stop at the first pivot.
        let g = [
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
        ];
        let x = solve(&g, &[1.0, 2.0, 3.0, -9.0]).unwrap();
        assert_eq!(x, [-2.0, 1.0, 1.0 / 3.0, -1.0]);
    }

    #[test]
    fn pivoting_handles_a_zero_on_the_diagonal() {
        // No elimination order without row exchanges survives this.
        let a = [[0.0, 1.0], [1.0, 0.0]];
        assert_eq!(solve(&a, &[3.0, 4.0]), Some([4.0, 3.0]));
        let a = [[0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 4.0]];
        assert_eq!(solve(&a, &[2.0, 3.0, 8.0]), Some([3.0, 1.0, 2.0]));
        let a = [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        assert_eq!(solve(&a, &[1.0, 2.0, 3.0, 4.0]), Some([4.0, 3.0, 2.0, 1.0]));
    }

    #[test]
    fn singular_and_non_finite_matrices_are_refused() {
        assert_eq!(solve(&[[0.0]], &[1.0]), None);
        assert_eq!(solve(&[[1.0, 2.0], [2.0, 4.0]], &[1.0, 1.0]), None);
        assert_eq!(
            solve(
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                &[1.0, 1.0, 1.0]
            ),
            None
        );
        let mut a = [[0.0; 5]; 5];
        for (i, row) in a.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        a[4] = a[3];
        assert!(Lu::factor(&a).is_none());
        assert_eq!(solve(&a, &[1.0; 5]), None);
        assert_eq!(solve(&[[f64::NAN, 0.0], [0.0, 1.0]], &[1.0, 1.0]), None);
        assert_eq!(solve(&[[f64::INFINITY]], &[1.0]), None);
        assert_eq!(solve(&[[0.0, 0.0], [0.0, 0.0]], &[0.0, 0.0]), None);
    }
}
