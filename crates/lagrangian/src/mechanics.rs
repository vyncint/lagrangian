//! The equations-of-motion engine.
//!
//! Everything here is a plain function of plain `f64` arrays. Nothing
//! integrates and nothing allocates: these are the pieces an ODE solver, a
//! game loop or a controller needs from a Lagrangian, and the crate's own
//! solvers are built from them and nothing else.

// Filling a matrix column by column indexes two arrays with one counter;
// the iterator form clippy prefers hides which index is the column.
#![allow(clippy::needless_range_loop)]

use core::array::from_fn;

use crate::ad::{Dual, Grad, Taylor2};
use crate::error::SimError;
use crate::linalg::{self, Lu};
use crate::math;
use crate::scalar::Scalar;
use crate::segment::State;

/// A mechanical system with `N` generalized coordinates, defined by its
/// Lagrangian `L(q, q̇, t) = T − V`.
///
/// Implement [`lagrangian`](Lagrangian::lagrangian) with ordinary arithmetic
/// on the generic scalar `S`. The engine calls it with dual numbers to get
/// the derivatives that make up the equations of motion; you never write
/// one.
///
/// ```
/// use lagrangian::prelude::*;
///
/// /// A mass on a spring.
/// struct Oscillator { m: f64, k: f64 }
///
/// impl Lagrangian<1> for Oscillator {
///     fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
///         dq[0].powi(2) * (0.5 * self.m) - q[0].powi(2) * (0.5 * self.k)
///     }
/// }
///
/// let sys = Oscillator { m: 2.0, k: 8.0 };
/// // m·q̈ = −k·q
/// assert_eq!(accelerations(&sys, [0.5], [0.0], 0.0), Ok([-2.0]));
/// ```
pub trait Lagrangian<const N: usize> {
    /// `L(q, q̇, t)`.
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> S;

    /// Generalized forces that do not come from the Lagrangian: friction,
    /// drag, a prescribed drive. Zero unless overridden.
    ///
    /// By the principle of virtual work, a Cartesian force `F` on a point
    /// `r(q)` contributes `Q_i = F · ∂r/∂q_i`.
    fn generalized_force<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> [S; N] {
        let _ = (q, dq, t);
        [S::ZERO; N]
    }
}

impl<L: Lagrangian<N> + ?Sized, const N: usize> Lagrangian<N> for &L {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> S {
        (**self).lagrangian(q, dq, t)
    }
    fn generalized_force<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> [S; N] {
        (**self).generalized_force(q, dq, t)
    }
}

/// A system with `M` external inputs: motor torques, thrust, a hand on the
/// cart.
///
/// The input enters as a generalized force, added to
/// [`Lagrangian::generalized_force`]. Wrap the system in [`WithInput`] to
/// integrate it with the input held constant, and use [`linearize`] for the
/// state-space matrices a controller is designed from.
pub trait Controlled<const N: usize, const M: usize>: Lagrangian<N> {
    /// The generalized force produced by the input `u`.
    fn input_force<S: Scalar>(&self, q: [S; N], dq: [S; N], u: [S; M], t: S) -> [S; N];
}

impl<L: Controlled<N, M> + ?Sized, const N: usize, const M: usize> Controlled<N, M> for &L {
    fn input_force<S: Scalar>(&self, q: [S; N], dq: [S; N], u: [S; M], t: S) -> [S; N] {
        (**self).input_force(q, dq, u, t)
    }
}

/// A controlled system with its input held at a fixed value: a plain
/// [`Lagrangian`], so every solver works on it unchanged.
///
/// The input is held constant across a step (zero-order hold). Change it
/// between steps through [`Stepper::system_mut`](crate::Stepper::system_mut):
///
/// ```
/// use lagrangian::prelude::*;
/// use lagrangian::systems::CartPole;
///
/// let cart = CartPole::default();
/// let mut live = Stepper::new(
///     WithInput { sys: &cart, u: [0.0] },
///     0.0, [0.0, 0.1], [0.0, 0.0],
///     SimConfig::default(),
/// )?;
/// live.step()?;
/// live.system_mut().u = [2.5];   // push the cart from now on
/// live.step()?;
/// # Ok::<(), SimError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WithInput<L, const M: usize> {
    /// The controlled system.
    pub sys: L,
    /// The input currently applied.
    pub u: [f64; M],
}

impl<L: Controlled<N, M>, const N: usize, const M: usize> Lagrangian<N> for WithInput<L, M> {
    fn lagrangian<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> S {
        self.sys.lagrangian(q, dq, t)
    }
    fn generalized_force<S: Scalar>(&self, q: [S; N], dq: [S; N], t: S) -> [S; N] {
        let own = self.sys.generalized_force(q, dq, t);
        let input = self.sys.input_force(q, dq, self.u.map(S::from_f64), t);
        from_fn(|i| own[i] + input[i])
    }
}

/// Every term of the Euler-Lagrange equations at one state:
///
/// ```text
/// M · q̈ = ∇_q L + Q − C
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EomTerms<const N: usize> {
    /// The Lagrangian itself, `L`.
    pub lagrangian: f64,
    /// The mass matrix `M_ij = ∂²L/∂q̇_i∂q̇_j`, symmetrized. Not necessarily
    /// positive definite.
    pub mass: [[f64; N]; N],
    /// `∇_q L`.
    pub grad_q: [f64; N],
    /// The generalized momentum `p = ∇_q̇ L`.
    pub momentum: [f64; N],
    /// The convective vector `C_i = Σ_j ∂²L/∂q̇_i∂q_j · q̇_j + ∂²L/∂q̇_i∂t`:
    /// Coriolis, centrifugal and explicit-time terms together.
    pub convective: [f64; N],
    /// The non-conservative generalized force `Q`.
    pub force: [f64; N],
    /// `∂L/∂t`, the explicit time dependence.
    pub dl_dt: f64,
}

impl<const N: usize> EomTerms<N> {
    /// The right-hand side `∇_q L + Q − C`.
    pub fn rhs(&self) -> [f64; N] {
        from_fn(|i| self.grad_q[i] + self.force[i] - self.convective[i])
    }
}

/// The total energy of a state and its split into kinetic and potential
/// parts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Energy {
    /// The Jacobi integral `E = p · q̇ − L`: the quantity an autonomous
    /// conservative system conserves, whatever its coordinates.
    pub total: f64,
    /// `E − V_eff`.
    pub kinetic: f64,
    /// The effective potential `V_eff = −L(q, 0, t)`.
    pub potential: f64,
}

/// The Lagrangian with `q̇` as the differentiation variables: `val` is `L`
/// and `d` is the momentum.
fn momentum_pass<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> Grad<N> {
    sys.lagrangian(
        q.map(Grad::constant),
        from_fn(|i| Grad::variable(dq[i], i)),
        Grad::constant(t),
    )
}

/// Every term of the equations of motion, in `N + 2` evaluations of the
/// Lagrangian.
///
/// One pass differentiates with respect to `q`. One pass carries the
/// gradient with respect to `q̇` and, on top of it, a single directional
/// derivative along `(δq, δq̇, δt) = (q̇, 0, 1)`, which is the whole
/// convective vector at once. `N` passes fill the columns of the mass
/// matrix. Each pass is `O(N)` wide, so the total work is
/// `O(N²)` evaluations' worth: a constant factor less than differentiating
/// entry by entry, not a different order.
pub fn eom_terms<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> EomTerms<N> {
    // Pass 1: gradient with respect to q.
    let by_q = sys.lagrangian(
        from_fn(|i| Grad::variable(q[i], i)),
        dq.map(Grad::constant),
        Grad::constant(t),
    );

    // Pass 2: gradient with respect to q̇ inside, and outside the derivative
    // along the flow. The direction is the *number* q̇, a constant as far as
    // the inner gradient is concerned: seeding it as a variable would add
    // ∇_q L to the result.
    let c = Grad::<N>::constant;
    let along_flow = sys.lagrangian(
        from_fn(|i| Dual::new(c(q[i]), c(dq[i]))),
        from_fn(|i| Dual::new(Grad::variable(dq[i], i), Grad::ZERO)),
        Dual::new(c(t), c(1.0)),
    );

    EomTerms {
        lagrangian: by_q.val,
        mass: mass_matrix(sys, q, dq, t),
        grad_q: by_q.d,
        momentum: along_flow.val.d,
        convective: along_flow.eps.d,
        force: sys.generalized_force(q, dq, t),
        dl_dt: along_flow.eps.val - dot(&by_q.d, &dq),
    }
}

/// The mass matrix `M_ij = ∂²L/∂q̇_i∂q̇_j`, symmetrized.
///
/// It is the Hessian of the Lagrangian in the velocities, whatever form the
/// kinetic energy takes. It is symmetric, and for a geodesic Lagrangian it
/// is the metric, so it may be indefinite.
pub fn mass_matrix<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> [[f64; N]; N] {
    let c = Grad::<N>::constant;
    let mut mass = [[0.0; N]; N];
    for j in 0..N {
        let column = sys.lagrangian(
            q.map(|x| Dual::constant(c(x))),
            from_fn(|i| Dual::new(Grad::variable(dq[i], i), c(if i == j { 1.0 } else { 0.0 }))),
            Dual::constant(c(t)),
        );
        for (i, row) in mass.iter_mut().enumerate() {
            row[j] = column.eps.d[i];
        }
    }
    symmetrize(&mut mass);
    mass
}

/// The two triangles of a Hessian agree in exact arithmetic and differ in
/// the last bits in floating point; averaging removes the difference before
/// it reaches the solver.
fn symmetrize<const N: usize>(m: &mut [[f64; N]; N]) {
    for i in 0..N {
        for j in 0..i {
            let mean = 0.5 * (m[i][j] + m[j][i]);
            m[i][j] = mean;
            m[j][i] = mean;
        }
    }
}

/// Why a linear solve against `matrix` failed: a matrix that is not finite
/// is a state that went wrong, not a singular mass matrix.
pub(crate) fn solve_failure<const N: usize>(matrix: &[[f64; N]; N], t: f64) -> SimError {
    if matrix.iter().flatten().all(|x| x.is_finite()) {
        SimError::SingularMassMatrix { t }
    } else {
        SimError::NonFiniteState { t }
    }
}

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}

/// The generalized accelerations `q̈`: the equations of motion solved for
/// their highest derivative.
///
/// This is the function to hand to any ODE solver as the right-hand side of
/// `d/dt (q, q̇) = (q̇, q̈)`.
///
/// # Errors
///
/// [`SimError::SingularMassMatrix`] if the mass matrix cannot be inverted,
/// and [`SimError::NonFiniteState`] if the mass matrix or the result is
/// `NaN` or infinite.
pub fn accelerations<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> Result<[f64; N], SimError> {
    let terms = eom_terms(sys, q, dq, t);
    let ddq =
        linalg::solve(&terms.mass, &terms.rhs()).ok_or_else(|| solve_failure(&terms.mass, t))?;
    if ddq.iter().all(|x| x.is_finite()) {
        Ok(ddq)
    } else {
        Err(SimError::NonFiniteState { t })
    }
}

/// The generalized momentum `p = ∇_q̇ L`.
pub fn momentum<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> [f64; N] {
    momentum_pass(sys, q, dq, t).d
}

/// The total energy by Legendre transform, `E = p · q̇ − L`, split into
/// kinetic and potential parts.
///
/// No assumption is made about the form of `L`. A rotating frame puts terms
/// without `q̇` into the kinetic energy and a magnetic field puts terms
/// linear in `q̇` into the Lagrangian; `½ q̇ᵀ M q̇ + V` is wrong for both, and
/// this is not.
pub fn energy<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> Energy {
    let pass = momentum_pass(sys, q, dq, t);
    let total = dot(&pass.d, &dq) - pass.val;
    let potential = -sys.lagrangian(q, [0.0; N], t);
    Energy {
        total,
        kinetic: total - potential,
        potential,
    }
}

/// The largest magnitude in a vector.
pub(crate) fn max_abs<const N: usize>(v: &[f64; N]) -> f64 {
    let mut largest = 0.0_f64;
    for &x in v {
        let m = math::abs(x);
        if m > largest {
            largest = m;
        }
    }
    largest
}

/// Decides when a Newton iteration has nothing left to gain.
///
/// The test is on the residual, relative to the size of the terms that were
/// summed to form it: once it is at the level of their rounding error, a
/// further update would be computed from noise. Also done when the residual
/// has stopped shrinking while already tiny: Newton converges quadratically
/// until it reaches the rounding floor and then wanders at that level, and a
/// floor raised by cancellation inside the Lagrangian itself is not visible
/// from here.
pub(crate) struct NewtonMonitor {
    previous: f64,
}

impl NewtonMonitor {
    pub(crate) const MAX_ITERATIONS: usize = 40;

    pub(crate) fn new() -> Self {
        Self {
            previous: f64::INFINITY,
        }
    }

    /// `residual` is the largest magnitude in the residual and `scale` the
    /// largest magnitude among the terms it was summed from.
    pub(crate) fn done(&mut self, residual: f64, scale: f64) -> bool {
        let converged = residual <= 16.0 * f64::EPSILON * scale;
        let stalled = residual >= 0.5 * self.previous && residual <= 1e-9 * scale;
        self.previous = residual;
        converged || stalled
    }
}

/// The velocity that has momentum `p`: the inverse Legendre transform,
/// solved by Newton's iteration with the mass matrix as its Jacobian.
///
/// For a Lagrangian quadratic in the velocities the first iteration is
/// already exact, from any guess.
///
/// # Errors
///
/// [`SimError::SingularMassMatrix`], [`SimError::NonFiniteState`] or
/// [`SimError::NewtonDidNotConverge`].
pub fn velocity_from_momentum<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    p: [f64; N],
    t: f64,
    dq_guess: [f64; N],
) -> Result<[f64; N], SimError> {
    let mut dq = dq_guess;
    let mut monitor = NewtonMonitor::new();
    for _ in 0..NewtonMonitor::MAX_ITERATIONS {
        let now = momentum(sys, q, dq, t);
        let residual: [f64; N] = from_fn(|i| p[i] - now[i]);
        if !residual.iter().all(|x| x.is_finite()) {
            return Err(SimError::NonFiniteState { t });
        }
        if monitor.done(max_abs(&residual), max_abs(&p).max(max_abs(&now))) {
            return Ok(dq);
        }
        let mass = mass_matrix(sys, q, dq, t);
        let update = linalg::solve(&mass, &residual).ok_or_else(|| solve_failure(&mass, t))?;
        for i in 0..N {
            dq[i] += update[i];
        }
    }
    Err(SimError::NewtonDidNotConverge { t })
}

/// The linearization of the equations of motion about a state: how the
/// accelerations respond to a small change in coordinates, velocities and
/// inputs.
///
/// With the state `x = (δq, δq̇)` the linear model is `ẋ = A x + B δu`, where
///
/// ```text
///     ⎡  0     I   ⎤        ⎡ 0 ⎤
/// A = ⎢            ⎥    B = ⎢   ⎥
///     ⎣ a_q   a_dq ⎦        ⎣ b ⎦
/// ```
///
/// The blocks are returned rather than the dense `2N × 2N` matrix because
/// stable Rust cannot name an array of length `2 * N`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Linearization<const N: usize, const M: usize> {
    /// `∂q̈_i/∂q_j`.
    pub a_q: [[f64; N]; N],
    /// `∂q̈_i/∂q̇_j`.
    pub a_dq: [[f64; N]; N],
    /// `∂q̈_i/∂u_k`.
    pub b: [[f64; M]; N],
}

#[cfg(feature = "alloc")]
impl<const N: usize, const M: usize> Linearization<N, M> {
    /// The dense state-space pair `(A, B)` in row-major order: `A` is
    /// `2N × 2N` and `B` is `2N × M`.
    pub fn to_dense(&self) -> (alloc::vec::Vec<f64>, alloc::vec::Vec<f64>) {
        let n2 = 2 * N;
        let mut a = alloc::vec![0.0; n2 * n2];
        let mut b = alloc::vec![0.0; n2 * M];
        for i in 0..N {
            a[i * n2 + N + i] = 1.0;
            for j in 0..N {
                a[(N + i) * n2 + j] = self.a_q[i][j];
                a[(N + i) * n2 + N + j] = self.a_dq[i][j];
            }
            for k in 0..M {
                b[(N + i) * M + k] = self.b[i][k];
            }
        }
        (a, b)
    }
}

/// The derivative of the mass matrix and of the right-hand side along one
/// direction `(δq, δq̇) = (a, b)` of state space.
///
/// The same passes as [`eom_terms`], each wrapped in one more [`Dual`]
/// whose derivative part carries the direction.
fn directional_terms<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
    a: [f64; N],
    b: [f64; N],
) -> ([[f64; N]; N], [f64; N]) {
    let c = Grad::<N>::constant;

    // δ(∇_q L).
    let by_q = sys.lagrangian(
        from_fn(|i| Dual::new(Grad::variable(q[i], i), c(a[i]))),
        from_fn(|i| Dual::new(c(dq[i]), c(b[i]))),
        Dual::constant(c(t)),
    );

    // δC. The inner pair is the seed of `eom_terms`' second pass; the outer
    // derivative is that seed's own change along (a, b) — including the
    // change of the convective direction q̇ itself, which is b.
    let along_flow = sys.lagrangian(
        from_fn(|i| Dual::new(Dual::new(c(q[i]), c(dq[i])), Dual::new(c(a[i]), c(b[i])))),
        from_fn(|i| {
            Dual::new(
                Dual::new(Grad::variable(dq[i], i), Grad::ZERO),
                Dual::constant(c(b[i])),
            )
        }),
        Dual::constant(Dual::new(c(t), c(1.0))),
    );

    // δM, column by column.
    let mut d_mass = [[0.0; N]; N];
    for j in 0..N {
        let column = sys.lagrangian(
            from_fn(|i| Dual::new(Dual::constant(c(q[i])), Dual::constant(c(a[i])))),
            from_fn(|i| {
                Dual::new(
                    Dual::new(Grad::variable(dq[i], i), c(if i == j { 1.0 } else { 0.0 })),
                    Dual::constant(c(b[i])),
                )
            }),
            Dual::constant(Dual::constant(c(t))),
        );
        for (i, row) in d_mass.iter_mut().enumerate() {
            row[j] = column.eps.eps.d[i];
        }
    }
    symmetrize(&mut d_mass);

    // δQ.
    let force = sys.generalized_force(
        from_fn(|i| Dual::new(q[i], a[i])),
        from_fn(|i| Dual::new(dq[i], b[i])),
        Dual::constant(t),
    );

    let d_rhs = from_fn(|i| by_q.eps.d[i] + force[i].eps - along_flow.eps.eps.d[i]);
    (d_mass, d_rhs)
}

/// The two state blocks of the linearization, and the factored mass matrix
/// for whoever needs to solve against it again.
struct StateBlocks<const N: usize> {
    a_q: [[f64; N]; N],
    a_dq: [[f64; N]; N],
    mass: Lu<N>,
}

fn linearize_state<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> Result<StateBlocks<N>, SimError> {
    let terms = eom_terms(sys, q, dq, t);
    let lu = Lu::factor(&terms.mass).ok_or_else(|| solve_failure(&terms.mass, t))?;
    let ddq = lu.solve(&terms.rhs());

    let mut a_q = [[0.0; N]; N];
    let mut a_dq = [[0.0; N]; N];
    for j in 0..N {
        let unit: [f64; N] = from_fn(|i| if i == j { 1.0 } else { 0.0 });
        for (block, a, b) in [(&mut a_q, unit, [0.0; N]), (&mut a_dq, [0.0; N], unit)] {
            // M q̈ = rhs, differentiated: M δq̈ = δrhs − δM q̈.
            let (d_mass, d_rhs) = directional_terms(sys, q, dq, t, a, b);
            let reduced: [f64; N] = from_fn(|i| d_rhs[i] - dot(&d_mass[i], &ddq));
            let column = lu.solve(&reduced);
            for i in 0..N {
                block[i][j] = column[i];
            }
        }
    }
    if a_q
        .iter()
        .chain(a_dq.iter())
        .flatten()
        .all(|x| x.is_finite())
    {
        Ok(StateBlocks {
            a_q,
            a_dq,
            mass: lu,
        })
    } else {
        Err(SimError::NonFiniteState { t })
    }
}

/// Linearizes a controlled system about the state `(q, q̇)` and input `u`.
///
/// The result is exact: it is computed by differentiating the equations of
/// motion once more, not by finite differences. At an equilibrium it is the
/// model LQR and pole placement are designed from.
///
/// ```
/// use lagrangian::prelude::*;
/// use lagrangian::systems::CartPole;
///
/// let cart = CartPole { m_cart: 1.0, m_pole: 0.1, length: 0.5, g: 9.81, ..Default::default() };
/// // Upright and at rest: an unstable equilibrium.
/// let lin = linearize(&cart, [0.0, 0.0], [0.0, 0.0], [0.0], 0.0)?;
/// // θ̈ = (M + m)·g/(M·l)·θ − u/(M·l)
/// assert!((lin.a_q[1][1] - 1.1 * 9.81 / 0.5).abs() < 1e-12);
/// assert!((lin.b[1][0] + 1.0 / 0.5).abs() < 1e-12);
/// # Ok::<(), SimError>(())
/// ```
///
/// # Errors
///
/// [`SimError::SingularMassMatrix`] or [`SimError::NonFiniteState`].
pub fn linearize<L: Controlled<N, M>, const N: usize, const M: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    u: [f64; M],
    t: f64,
) -> Result<Linearization<N, M>, SimError> {
    let held = WithInput { sys, u };
    let StateBlocks {
        a_q,
        a_dq,
        mass: lu,
    } = linearize_state(&held, q, dq, t)?;

    // Only the input force depends on u, so ∂q̈/∂u_k = M⁻¹ ∂Q/∂u_k.
    let mut b = [[0.0; M]; N];
    for k in 0..M {
        let force = sys.input_force(
            q.map(Dual::constant),
            dq.map(Dual::constant),
            from_fn(|j| Dual::new(u[j], if j == k { 1.0 } else { 0.0 })),
            Dual::constant(t),
        );
        let column = lu.solve(&force.map(|f| f.eps));
        for i in 0..N {
            b[i][k] = column[i];
        }
    }
    if b.iter().flatten().all(|x| x.is_finite()) {
        Ok(Linearization { a_q, a_dq, b })
    } else {
        Err(SimError::NonFiniteState { t })
    }
}

/// Linearizes a system without inputs about the state `(q, q̇)`: the `a_q`
/// and `a_dq` blocks of [`Linearization`], with an empty `b`.
///
/// The eigenvalues of the resulting `A` say whether an equilibrium is
/// stable and give the frequencies of small oscillations about it.
///
/// # Errors
///
/// [`SimError::SingularMassMatrix`] or [`SimError::NonFiniteState`].
pub fn linearize_free<L: Lagrangian<N> + ?Sized, const N: usize>(
    sys: &L,
    q: [f64; N],
    dq: [f64; N],
    t: f64,
) -> Result<Linearization<N, 0>, SimError> {
    let StateBlocks { a_q, a_dq, .. } = linearize_state(sys, q, dq, t)?;
    Ok(Linearization {
        a_q,
        a_dq,
        b: [[]; N],
    })
}

/// A map from generalized coordinates to `P` points in `D`-dimensional
/// space: where the bobs, the cart and the bead actually are.
///
/// Write positions only. [`kinematics`] differentiates the map to get the
/// velocity and acceleration of every point.
pub trait SceneMap<const N: usize, const D: usize, const P: usize = 1> {
    /// The positions of the points at coordinates `q` and time `t`.
    fn points<S: Scalar>(&self, q: [S; N], t: S) -> [[S; D]; P];
}

impl<T: SceneMap<N, D, P> + ?Sized, const N: usize, const D: usize, const P: usize>
    SceneMap<N, D, P> for &T
{
    fn points<S: Scalar>(&self, q: [S; N], t: S) -> [[S; D]; P] {
        (**self).points(q, t)
    }
}

/// The position, velocity and acceleration of one point in space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Kinematics<const D: usize> {
    /// Position.
    pub pos: [f64; D],
    /// Velocity.
    pub vel: [f64; D],
    /// Acceleration.
    pub acc: [f64; D],
}

/// The position, velocity and acceleration of every point of a
/// [`SceneMap`] at a state, in one evaluation of the map.
///
/// Each coordinate enters as the polynomial `q + q̇·τ + ½q̈·τ²`; what comes
/// out is the same expansion of each point. Centripetal, tangential,
/// Coriolis and moving-pivot accelerations are all in it, because they are
/// all in the chain rule.
pub fn kinematics<Map, const N: usize, const D: usize, const P: usize>(
    map: &Map,
    state: &State<N>,
) -> [Kinematics<D>; P]
where
    Map: SceneMap<N, D, P> + ?Sized,
{
    let q = from_fn(|i| Taylor2::new(state.q[i], state.dq[i], 0.5 * state.ddq[i]));
    let points = map.points(q, Taylor2::new(state.t, 1.0, 0.0));
    points.map(|p| Kinematics {
        pos: p.map(|x| x.c0),
        vel: p.map(|x| x.c1),
        acc: p.map(|x| 2.0 * x.c2),
    })
}

/// The position and velocity of a point given as a function of the
/// coordinates and time, for use **inside** a Lagrangian.
///
/// Write where the point is; get back how fast it moves, with the chain rule
/// applied for you. A kinetic energy is then `½ m (vx² + vy²)` with no
/// expansion by hand, however the pivot moves.
///
/// ```
/// use lagrangian::prelude::*;
///
/// /// A pendulum whose pivot is shaken vertically.
/// struct Shaken { l: f64, a: f64, nu: f64, g: f64 }
///
/// impl Lagrangian<1> for Shaken {
///     fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], t: S) -> S {
///         let (pos, vel) = velocity_of(
///             |q, t| [
///                 q[0].sin() * self.l,
///                 (t * self.nu).cos() * self.a - q[0].cos() * self.l,
///             ],
///             q, dq, t,
///         );
///         (vel[0].powi(2) + vel[1].powi(2)) * 0.5 - pos[1] * self.g
///     }
/// }
/// # let sys = Shaken { l: 1.0, a: 0.1, nu: 30.0, g: 9.81 };
/// # assert!(accelerations(&sys, [0.3], [0.0], 0.0).is_ok());
/// ```
pub fn velocity_of<S: Scalar, const N: usize, const D: usize, F>(
    f: F,
    q: [S; N],
    dq: [S; N],
    t: S,
) -> ([S; D], [S; D])
where
    F: FnOnce([Dual<S>; N], Dual<S>) -> [Dual<S>; D],
{
    let moving = f(from_fn(|i| Dual::new(q[i], dq[i])), Dual::new(t, S::ONE));
    (moving.map(|x| x.val), moving.map(|x| x.eps))
}
