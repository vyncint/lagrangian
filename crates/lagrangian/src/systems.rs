//! Ready-made systems: the benchmarks the crate is verified against, and
//! starting points for your own.
//!
//! Each one is a few lines of Lagrangian and nothing else. No equation of
//! motion is written anywhere in this file.

use crate::mechanics::{Controlled, Lagrangian, SceneMap, velocity_of};
use crate::scalar::Scalar;

/// Two point masses on massless rods, the second hanging from the first.
///
/// Coordinates are the angles of the two rods from the downward vertical.
/// Regular for small swings and chaotic for large ones; the mass matrix
/// depends on the configuration, so the system is not separable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoublePendulum {
    /// Mass of the first bob.
    pub m1: f64,
    /// Mass of the second bob.
    pub m2: f64,
    /// Length of the first rod.
    pub l1: f64,
    /// Length of the second rod.
    pub l2: f64,
    /// Gravitational acceleration.
    pub g: f64,
}

impl Default for DoublePendulum {
    fn default() -> Self {
        Self {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
        }
    }
}

impl Lagrangian<2> for DoublePendulum {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
        let [th1, th2] = q;
        let [w1, w2] = dq;
        let v1_sq = (w1 * self.l1).powi(2);
        let v2_sq = v1_sq
            + (w2 * self.l2).powi(2)
            + (w1 * w2 * (th1 - th2).cos()) * (2.0 * self.l1 * self.l2);

        let kinetic = v1_sq * (0.5 * self.m1) + v2_sq * (0.5 * self.m2);
        let y1 = -th1.cos() * self.l1;
        let y2 = y1 - th2.cos() * self.l2;
        let potential = y1 * (self.m1 * self.g) + y2 * (self.m2 * self.g);

        kinetic - potential
    }
}

/// The two bobs, with the pivot at the origin and `y` up.
impl SceneMap<2, 2, 2> for DoublePendulum {
    fn points<S: Scalar>(&self, q: [S; 2], _t: S) -> [[S; 2]; 2] {
        let (s1, c1) = q[0].sin_cos();
        let (s2, c2) = q[1].sin_cos();
        let first = [s1 * self.l1, -c1 * self.l1];
        [first, [first[0] + s2 * self.l2, first[1] - c2 * self.l2]]
    }
}

/// A bead sliding without friction on a circular hoop that spins about its
/// vertical diameter at a fixed rate.
///
/// The coordinate is the angle from the bottom of the hoop. Above the
/// critical rate `Ω² = g/R` the bottom becomes unstable and two new
/// equilibria appear at `cos θ* = g/(R·Ω²)`.
///
/// The Lagrangian has a term without any velocity in it,
/// `½ m R² Ω² sin²θ`, that belongs to the kinetic energy. `½ q̇ᵀMq̇ + V`
/// gets this system's energy wrong; the Jacobi integral the crate computes
/// is what is conserved.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeadOnRotatingHoop {
    /// Mass of the bead.
    pub m: f64,
    /// Radius of the hoop.
    pub radius: f64,
    /// Angular velocity of the hoop.
    pub omega: f64,
    /// Gravitational acceleration.
    pub g: f64,
}

impl Default for BeadOnRotatingHoop {
    fn default() -> Self {
        Self {
            m: 1.0,
            radius: 1.0,
            omega: 5.0,
            g: 9.81,
        }
    }
}

impl BeadOnRotatingHoop {
    /// The angle of the off-axis equilibria, `acos(g/(R·Ω²))`, or `None`
    /// when the hoop spins too slowly for them to exist.
    pub fn equilibrium_angle(&self) -> Option<f64> {
        let c = self.g / (self.radius * self.omega * self.omega);
        (c < 1.0).then(|| Scalar::acos(c))
    }
}

impl Lagrangian<1> for BeadOnRotatingHoop {
    fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], _t: S) -> S {
        let (sin, cos) = q[0].sin_cos();
        let r = self.radius;
        let speed_sq = dq[0].powi(2) * (r * r) + sin.powi(2) * (r * r * self.omega * self.omega);
        speed_sq * (0.5 * self.m) + cos * (self.m * self.g * r)
    }
}

/// The bead in the laboratory frame, with `z` up and the hoop's centre at
/// the origin.
impl SceneMap<1, 3, 1> for BeadOnRotatingHoop {
    fn points<S: Scalar>(&self, q: [S; 1], t: S) -> [[S; 3]; 1] {
        let (sin, cos) = q[0].sin_cos();
        let (spin_sin, spin_cos) = (t * self.omega).sin_cos();
        let r = self.radius;
        [[sin * spin_cos * r, sin * spin_sin * r, -cos * r]]
    }
}

/// A pendulum whose pivot is shaken vertically: Kapitza's pendulum.
///
/// The coordinate is the angle from the downward vertical, so `π` is
/// straight up. Shaken fast enough, with `a²ν² > 2·g·l` for amplitude `a`
/// and angular frequency `ν`, the upright position becomes stable.
///
/// The Lagrangian is written from the position of the bob alone;
/// [`velocity_of`] supplies its velocity, moving pivot included.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KapitzaPendulum {
    /// Mass of the bob.
    pub m: f64,
    /// Length of the rod.
    pub length: f64,
    /// Gravitational acceleration.
    pub g: f64,
    /// Amplitude of the pivot's vertical motion.
    pub amplitude: f64,
    /// Angular frequency of the pivot's vertical motion.
    pub frequency: f64,
}

impl Default for KapitzaPendulum {
    fn default() -> Self {
        Self {
            m: 1.0,
            length: 1.0,
            g: 9.81,
            amplitude: 0.05,
            frequency: 200.0,
        }
    }
}

impl KapitzaPendulum {
    /// `a²ν² / (2·g·l)`: the upright position is stable when this exceeds
    /// one (for a small amplitude).
    pub fn stability_ratio(&self) -> f64 {
        (self.amplitude * self.frequency).powi(2) / (2.0 * self.g * self.length)
    }

    fn bob<S: Scalar>(&self, theta: S, t: S) -> [S; 2] {
        let (sin, cos) = theta.sin_cos();
        [
            sin * self.length,
            (t * self.frequency).cos() * self.amplitude - cos * self.length,
        ]
    }
}

impl Lagrangian<1> for KapitzaPendulum {
    fn lagrangian<S: Scalar>(&self, q: [S; 1], dq: [S; 1], t: S) -> S {
        let (pos, vel) = velocity_of(|q, t| self.bob(q[0], t), q, dq, t);
        (vel[0].powi(2) + vel[1].powi(2)) * (0.5 * self.m) - pos[1] * (self.m * self.g)
    }
}

/// The pivot and the bob, with `y` up.
impl SceneMap<1, 2, 2> for KapitzaPendulum {
    fn points<S: Scalar>(&self, q: [S; 1], t: S) -> [[S; 2]; 2] {
        [
            [S::ZERO, (t * self.frequency).cos() * self.amplitude],
            self.bob(q[0], t),
        ]
    }
}

/// A cart on a rail with a pole hinged to it: the standard control
/// benchmark.
///
/// Coordinates are the cart's position and the pole's angle **from
/// upright**, so `θ = 0` is the balancing point and the pole's tip is at
/// `(x + l·sin θ, l·cos θ)`. The one input is a horizontal force on the
/// cart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartPole {
    /// Mass of the cart.
    pub m_cart: f64,
    /// Mass at the tip of the pole.
    pub m_pole: f64,
    /// Length of the pole.
    pub length: f64,
    /// Gravitational acceleration.
    pub g: f64,
    /// Viscous friction on the cart, as a force per unit velocity.
    pub cart_damping: f64,
    /// Viscous friction at the hinge, as a torque per unit angular velocity.
    pub pole_damping: f64,
}

impl Default for CartPole {
    fn default() -> Self {
        Self {
            m_cart: 1.0,
            m_pole: 0.1,
            length: 0.5,
            g: 9.81,
            cart_damping: 0.0,
            pole_damping: 0.0,
        }
    }
}

impl CartPole {
    fn tip<S: Scalar>(&self, x: S, theta: S) -> [S; 2] {
        let (sin, cos) = theta.sin_cos();
        [x + sin * self.length, cos * self.length]
    }
}

impl Lagrangian<2> for CartPole {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], t: S) -> S {
        let (tip, tip_vel) = velocity_of(|q, _t| self.tip(q[0], q[1]), q, dq, t);
        dq[0].powi(2) * (0.5 * self.m_cart)
            + (tip_vel[0].powi(2) + tip_vel[1].powi(2)) * (0.5 * self.m_pole)
            - tip[1] * (self.m_pole * self.g)
    }

    fn generalized_force<S: Scalar>(&self, _q: [S; 2], dq: [S; 2], _t: S) -> [S; 2] {
        [dq[0] * -self.cart_damping, dq[1] * -self.pole_damping]
    }
}

impl Controlled<2, 1> for CartPole {
    fn input_force<S: Scalar>(&self, _q: [S; 2], _dq: [S; 2], u: [S; 1], _t: S) -> [S; 2] {
        [u[0], S::ZERO]
    }
}

/// The cart and the tip of the pole, with `y` up.
impl SceneMap<2, 2, 2> for CartPole {
    fn points<S: Scalar>(&self, q: [S; 2], _t: S) -> [[S; 2]; 2] {
        [[q[0], S::ZERO], self.tip(q[0], q[1])]
    }
}

/// A body in an inverse-square central field, in plane polar coordinates
/// `(r, θ)`.
///
/// The Lagrangian does not depend on `θ`, so the angular momentum
/// `p_θ = m·r²·θ̇` is conserved: the textbook cyclic coordinate, and the
/// test of a symplectic integrator's discrete Noether theorem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerOrbit {
    /// The gravitational parameter `G·M` of the central body.
    pub mu: f64,
    /// Mass of the orbiting body.
    pub m: f64,
}

impl Default for KeplerOrbit {
    fn default() -> Self {
        Self { mu: 1.0, m: 1.0 }
    }
}

impl Lagrangian<2> for KeplerOrbit {
    fn lagrangian<S: Scalar>(&self, q: [S; 2], dq: [S; 2], _t: S) -> S {
        let r = q[0];
        (dq[0].powi(2) + (r * dq[1]).powi(2)) * (0.5 * self.m) + r.recip() * (self.m * self.mu)
    }
}

/// The body, with the central mass at the origin.
impl SceneMap<2, 2, 1> for KeplerOrbit {
    fn points<S: Scalar>(&self, q: [S; 2], _t: S) -> [[S; 2]; 1] {
        let (sin, cos) = q[1].sin_cos();
        [[q[0] * cos, q[0] * sin]]
    }
}

/// A free particle or a photon around a non-rotating black hole: geodesics
/// of the Schwarzschild metric.
///
/// Coordinates are `(t, r, θ, φ)` and the independent variable is the
/// proper time of the particle, or an affine parameter for light. The
/// Lagrangian is half the squared length of the four-velocity,
///
/// ```text
/// L = ½·[ −(1 − r_s/r)·ṫ² + ṙ²/(1 − r_s/r) + r²·(θ̇² + sin²θ·φ̇²) ]
/// ```
///
/// in units with `c = 1`. Its "mass matrix" is the metric, with signature
/// `(−, +, +, +)`: symmetric and **indefinite**, which is why the crate's
/// linear solver pivots instead of assuming positive definiteness.
///
/// `t` and `φ` are cyclic, so `p_t` (minus the energy per unit mass) and
/// `p_φ` (the angular momentum per unit mass) are conserved, and so is `L`
/// itself: `−½` for a massive particle, `0` for light.
///
/// The coordinates are singular at the horizon `r = r_s` and at the poles
/// `θ = 0, π`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildGeodesic {
    /// The Schwarzschild radius `r_s = 2·G·M/c²`.
    pub rs: f64,
}

impl Default for SchwarzschildGeodesic {
    fn default() -> Self {
        Self { rs: 1.0 }
    }
}

impl SchwarzschildGeodesic {
    /// The four-velocity `(ṫ, ṙ, θ̇, φ̇)` of a massive particle in the
    /// equatorial plane at radius `r`, given its radial and angular rates
    /// with respect to proper time. `ṫ` is fixed by the normalization
    /// `g(u, u) = −1`.
    pub fn timelike_velocity(&self, r: f64, dr: f64, dphi: f64) -> [f64; 4] {
        let f = 1.0 - self.rs / r;
        let dt = Scalar::sqrt((1.0 + dr * dr / f + r * r * dphi * dphi) / f);
        [dt, dr, 0.0, dphi]
    }

    /// The tangent `(ṫ, ṙ, θ̇, φ̇)` of a light ray in the equatorial plane at
    /// radius `r`, given its radial and angular rates with respect to the
    /// affine parameter. `ṫ` is fixed by `g(k, k) = 0`.
    pub fn null_velocity(&self, r: f64, dr: f64, dphi: f64) -> [f64; 4] {
        let f = 1.0 - self.rs / r;
        let dt = Scalar::sqrt((dr * dr / f + r * r * dphi * dphi) / f);
        [dt, dr, 0.0, dphi]
    }

    /// The radius of the photon sphere, `1.5·r_s`: the one circular orbit
    /// light can follow, and an unstable one.
    pub fn photon_sphere(&self) -> f64 {
        1.5 * self.rs
    }
}

impl Lagrangian<4> for SchwarzschildGeodesic {
    fn lagrangian<S: Scalar>(&self, q: [S; 4], dq: [S; 4], _t: S) -> S {
        let [_, r, theta, _] = q;
        let [dt, dr, dtheta, dphi] = dq;
        let f = S::ONE - r.recip() * self.rs;
        let angular = dtheta.powi(2) + (theta.sin() * dphi).powi(2);
        (dr.powi(2) / f - f * dt.powi(2) + r.powi(2) * angular) * 0.5
    }
}

/// The particle in Cartesian-like coordinates, with the hole at the origin.
impl SceneMap<4, 3, 1> for SchwarzschildGeodesic {
    fn points<S: Scalar>(&self, q: [S; 4], _t: S) -> [[S; 3]; 1] {
        let [_, r, theta, phi] = q;
        let (st, ct) = theta.sin_cos();
        let (sp, cp) = phi.sin_cos();
        [[r * st * cp, r * st * sp, r * ct]]
    }
}
