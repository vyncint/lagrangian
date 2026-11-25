//! Small dense linear solves on the stack.
//!
//! The mass matrix of a mechanical system is symmetric, but it is **not**
//! always positive definite: the "mass matrix" of a geodesic Lagrangian is
//! the metric, and a spacetime metric has one negative eigenvalue. So there
//! is no Cholesky here. Everything is Gaussian elimination with partial
//! pivoting, which does not care about the sign of a pivot.
//!
//! Before elimination every row, and then every column, is multiplied by
//! the power of two that brings its largest entry into `[1, 2)`. That adds
//! no rounding of its own, and it lets a pivot be judged against a fixed
//! fraction of one. A matrix whose rows or columns differ in scale by many
//! orders of magnitude, such as a metric near a horizon or a system that
//! mixes kilograms with micrograms, is solved rather than mistaken for a
//! singular one.

use crate::math;

/// A pivot at or below this size is treated as zero. In the equilibrated
/// matrix every row and every column has its largest entry in `[1, 2)`.
const SINGULAR_PIVOT: f64 = 64.0 * f64::EPSILON;

/// Whether a pivot is distinguishable from zero. Written so that a `NaN`
/// fails.
#[inline]
fn is_pivot(pivot: f64) -> bool {
    math::abs(pivot) > SINGULAR_PIVOT
}

/// The power of two at or below `x`, or `None` if `x` is zero, subnormal or
/// not finite. Clearing the significand leaves the exponent, exactly.
#[inline]
fn power_of_two_below(x: f64) -> Option<f64> {
    if !(f64::MIN_POSITIVE..=f64::MAX).contains(&x) {
        return None;
    }
    Some(f64::from_bits(x.to_bits() & 0x7ff0_0000_0000_0000))
}

/// `a` with every row and then every column multiplied by a power of two,
/// and the factors: `R·a·C`, with `R = diag(rows)` and `C = diag(cols)`.
///
/// `None` if an entry is not finite, or a row or column has no entry in the
/// normal range: then the matrix is singular to working precision. 0.1.0
/// judged every pivot against the largest entry of the whole matrix instead,
/// and refused `diag(1e300, 1e-300)` (#16).
#[allow(clippy::type_complexity)]
fn equilibrate<const N: usize>(a: &[[f64; N]; N]) -> Option<([[f64; N]; N], [f64; N], [f64; N])> {
    let mut m = *a;
    let mut rows = [1.0; N];
    for (row, factor) in m.iter_mut().zip(rows.iter_mut()) {
        let mut largest = 0.0_f64;
        for &x in row.iter() {
            if !x.is_finite() {
                return None;
            }
            largest = largest.max(math::abs(x));
        }
        *factor = 1.0 / power_of_two_below(largest)?;
        for x in row.iter_mut() {
            *x *= *factor;
        }
    }
    let mut cols = [1.0; N];
    for (j, factor) in cols.iter_mut().enumerate() {
        let largest = m.iter().fold(0.0_f64, |l, row| l.max(math::abs(row[j])));
        *factor = 1.0 / power_of_two_below(largest)?;
        for row in m.iter_mut() {
            row[j] *= *factor;
        }
    }
    Some((m, rows, cols))
}

/// An `N × N` matrix factored as `P·R·A·C = L·U`: `R` and `C` are the
/// diagonal powers of two that equilibrate it, and `P` the row exchanges of
/// partial pivoting.
///
/// Factor once, then [`solve`](Lu::solve) for as many right-hand sides as
/// needed.
#[derive(Debug, Clone, Copy)]
pub struct Lu<const N: usize> {
    /// `L` below the diagonal (unit diagonal implied), `U` on and above it.
    lu: [[f64; N]; N],
    /// Row `i` of the factorization is row `perm[i]` of the original.
    perm: [usize; N],
    /// The diagonals of `R` and `C`.
    rows: [f64; N],
    cols: [f64; N],
}

impl<const N: usize> Lu<N> {
    /// Factors `a`, or returns `None` if it is singular to working precision
    /// or contains a value that is not finite.
    pub fn factor(a: &[[f64; N]; N]) -> Option<Self> {
        let (mut lu, rows, cols) = equilibrate(a)?;
        let mut perm: [usize; N] = core::array::from_fn(|i| i);

        for k in 0..N {
            let mut pivot_row = k;
            for r in (k + 1)..N {
                if math::abs(lu[r][k]) > math::abs(lu[pivot_row][k]) {
                    pivot_row = r;
                }
            }
            if pivot_row != k {
                lu.swap(k, pivot_row);
                perm.swap(k, pivot_row);
            }
            if !is_pivot(lu[k][k]) {
                return None;
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
        Some(Self {
            lu,
            perm,
            rows,
            cols,
        })
    }

    /// Solves `A x = b`.
    pub fn solve(&self, b: &[f64; N]) -> [f64; N] {
        let mut x: [f64; N] = core::array::from_fn(|i| b[self.perm[i]] * self.rows[self.perm[i]]);
        // L y = P R b
        for i in 0..N {
            for j in 0..i {
                x[i] -= self.lu[i][j] * x[j];
            }
        }
        // U z = y
        for i in (0..N).rev() {
            for j in (i + 1)..N {
                x[i] -= self.lu[i][j] * x[j];
            }
            x[i] /= self.lu[i][i];
        }
        // x = C z
        for (xi, c) in x.iter_mut().zip(self.cols) {
            *xi *= c;
        }
        x
    }
}

/// Solves `A x = b`, or returns `None` if `A` is singular to working
/// precision or contains a value that is not finite.
///
/// `N = 1, 2, 3` take fixed-size paths with no permutation bookkeeping;
/// every other size goes through [`Lu`]. All of them equilibrate and pivot,
/// so an indefinite matrix is as welcome as a positive definite one, and a
/// badly scaled one as a well scaled one.
pub fn solve<const N: usize>(a: &[[f64; N]; N], b: &[f64; N]) -> Option<[f64; N]> {
    if N > 3 {
        return Lu::factor(a).map(|lu| lu.solve(b));
    }
    let (m, rows, cols) = equilibrate(a)?;
    let rb: [f64; N] = core::array::from_fn(|i| b[i] * rows[i]);
    let mut z = [0.0; N];
    match N {
        0 => {}
        1 => z[0] = rb[0] / m[0][0],
        2 => z[..2].copy_from_slice(&solve2(
            [[m[0][0], m[0][1]], [m[1][0], m[1][1]]],
            [rb[0], rb[1]],
        )?),
        _ => z[..3].copy_from_slice(&solve3(
            [
                [m[0][0], m[0][1], m[0][2]],
                [m[1][0], m[1][1], m[1][2]],
                [m[2][0], m[2][1], m[2][2]],
            ],
            [rb[0], rb[1], rb[2]],
        )?),
    }
    Some(core::array::from_fn(|i| z[i] * cols[i]))
}

/// Partial pivoting on an equilibrated 2 × 2 matrix.
fn solve2(a: [[f64; 2]; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let (r0, r1, b0, b1) = if math::abs(a[0][0]) >= math::abs(a[1][0]) {
        (a[0], a[1], b[0], b[1])
    } else {
        (a[1], a[0], b[1], b[0])
    };
    if !is_pivot(r0[0]) {
        return None;
    }
    let f = r1[0] / r0[0];
    let u11 = r1[1] - f * r0[1];
    if !is_pivot(u11) {
        return None;
    }
    let x1 = (b1 - f * b0) / u11;
    let x0 = (b0 - r0[1] * x1) / r0[0];
    Some([x0, x1])
}

/// Partial pivoting on an equilibrated 3 × 3 matrix.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
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
    if !is_pivot(a[0][0]) {
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
    if !is_pivot(a[1][1]) {
        return None;
    }
    let f = a[2][1] / a[1][1];
    a[2][2] -= f * a[1][2];
    b[2] -= f * b[1];
    if !is_pivot(a[2][2]) {
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
    fn badly_scaled_but_invertible_matrices_are_solved() {
        // 0.1.0 compared every pivot with the largest entry of the whole
        // matrix, so entries spanning more than about 7e13 were called
        // singular (#16). Rows and columns are now equilibrated first.
        assert_eq!(solve(&[[1e-300]], &[1.0]), Some([1.0 / 1e-300]));
        assert_eq!(
            solve(&[[1e300, 0.0], [0.0, 1e-300]], &[1.0, 1.0]),
            Some([1.0 / 1e300, 1.0 / 1e-300])
        );
        assert_eq!(
            solve(
                &[[1e200, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-200]],
                &[1.0, 1.0, 1.0]
            ),
            Some([1.0 / 1e200, 1.0, 1.0 / 1e-200])
        );
        // The metric just outside a horizon, `diag(-f, 1/f, r², r²)`.
        let f = 1e-12;
        let metric = [
            [-f, 0.0, 0.0, 0.0],
            [0.0, 1.0 / f, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let x = solve(&metric, &[1.0, 1.0, 1.0, 1.0]).expect("invertible");
        assert_eq!(x, [1.0 / -f, 1.0 / (1.0 / f), 1.0, 1.0]);

        // Scaling the rows of a system does not change its solution.
        fn scaled<const N: usize>(scales: [f64; N]) {
            let mut rng = Lcg(0xfeed + N as u64);
            for _ in 0..50 {
                let a = symmetric_indefinite::<N>(&mut rng);
                let b: [f64; N] = core::array::from_fn(|_| rng.next() * 3.0);
                let plain = solve(&a, &b).expect("well conditioned");
                let sa: [[f64; N]; N] = core::array::from_fn(|i| a[i].map(|x| x * scales[i]));
                let sb: [f64; N] = core::array::from_fn(|i| b[i] * scales[i]);
                let x = solve(&sa, &sb).expect("a row-scaled invertible matrix");
                for i in 0..N {
                    let scale = 1.0_f64.max(plain[i].abs());
                    assert!(
                        (x[i] - plain[i]).abs() <= 1e-12 * scale,
                        "N = {N}: {x:?} vs {plain:?}"
                    );
                }
            }
        }
        scaled([1e150, 1e-150]);
        scaled([1e-150, 1.0, 1e150]);
        scaled([1e150, 1.0, 1e-150, 1e100]);
        scaled([1e-200, 1e200, 1.0, 1e-100, 1e100, 1.0]);

        // Nor does scaling rows and columns alike, `D·A·D`: the mass matrix
        // in coordinates of very different units. Every pivot is then tiny
        // next to its own row, which pivoting on rows alone calls singular.
        fn congruent<const N: usize>(d: [f64; N]) {
            let mut rng = Lcg(0xbeef + N as u64);
            for _ in 0..50 {
                let a = symmetric_indefinite::<N>(&mut rng);
                let b: [f64; N] = core::array::from_fn(|_| rng.next() * 3.0);
                let plain = solve(&a, &b).expect("well conditioned");
                let da: [[f64; N]; N] =
                    core::array::from_fn(|i| core::array::from_fn(|j| d[i] * a[i][j] * d[j]));
                let db: [f64; N] = core::array::from_fn(|i| d[i] * b[i]);
                let y = solve(&da, &db).expect("a congruent invertible matrix");
                for i in 0..N {
                    let scale = 1.0_f64.max(plain[i].abs());
                    assert!(
                        (y[i] * d[i] - plain[i]).abs() <= 1e-12 * scale,
                        "N = {N}: {y:?} vs {plain:?}"
                    );
                }
            }
        }
        congruent([1e-20, 1.0]);
        congruent([1e-20, 1.0, 1e20]);
        congruent([1e60, 1.0, 1e-60, 1e30]);
        congruent([1e-40, 1e40, 1.0, 1e-20, 1e20]);
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
