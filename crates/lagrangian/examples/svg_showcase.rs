//! Publication-style figures from three systems, with no dependencies.
//!
//! ```sh
//! cargo run --release --example svg_showcase            # writes target/svg_showcase/
//! cargo run --release --example svg_showcase -- out/    # or anywhere else
//! ```
//!
//! Each figure is written in a dark and a light theme. Everything drawn is
//! read off a `Trajectory`: positions, velocity and acceleration arrows
//! through `SceneMap`, phase portraits through `phase_point`, Poincaré
//! sections through `crossings`, and energy through `energy`.

use std::error::Error;
use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use lagrangian::prelude::*;
use lagrangian::systems::{BeadOnRotatingHoop, DoublePendulum, SchwarzschildGeodesic};

type Point = (f64, f64);

// ---------------------------------------------------------------------------
// A very small SVG plotter
// ---------------------------------------------------------------------------

struct Theme {
    name: &'static str,
    background: &'static str,
    panel: &'static str,
    text: &'static str,
    muted: &'static str,
    grid: &'static str,
    /// Series colours, in the order they are used.
    series: [&'static str; 5],
    velocity: &'static str,
    acceleration: &'static str,
}

const DARK: Theme = Theme {
    name: "dark",
    background: "#0e1116",
    panel: "#161b22",
    text: "#e6edf3",
    muted: "#8b949e",
    grid: "#262c36",
    series: ["#58a6ff", "#f0883e", "#bc8cff", "#3fb950", "#f778ba"],
    velocity: "#3fb950",
    acceleration: "#ff7b72",
};

const LIGHT: Theme = Theme {
    name: "light",
    background: "#ffffff",
    panel: "#f6f8fa",
    text: "#1f2328",
    muted: "#656d76",
    grid: "#d8dee4",
    series: ["#0969da", "#bc4c00", "#8250df", "#1a7f37", "#bf3989"],
    velocity: "#1a7f37",
    acceleration: "#cf222e",
};

struct Figure<'a> {
    svg: String,
    theme: &'a Theme,
    clips: usize,
}

/// A plotting area: where it is on the page, and what data range it shows.
#[derive(Clone, Copy)]
struct Axes {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
    x: (f64, f64),
    y: (f64, f64),
    clip: usize,
}

impl Axes {
    fn to_page(self, (x, y): Point) -> Point {
        (
            self.left + (x - self.x.0) / (self.x.1 - self.x.0) * self.width,
            self.top + self.height - (y - self.y.0) / (self.y.1 - self.y.0) * self.height,
        )
    }
}

/// Round tick positions covering `[lo, hi]`.
fn ticks(lo: f64, hi: f64) -> Vec<f64> {
    let rough = (hi - lo) / 5.0;
    let magnitude = 10f64.powf(rough.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * magnitude)
        .find(|s| *s >= rough)
        .unwrap_or(10.0 * magnitude);
    let mut out = Vec::new();
    let mut tick = (lo / step).ceil() * step;
    while tick <= hi + 1e-9 * step {
        out.push(if tick.abs() < 1e-12 * step { 0.0 } else { tick });
        tick += step;
    }
    out
}

fn label(value: f64) -> String {
    let text = format!("{value:.3}");
    let text = text.trim_end_matches('0').trim_end_matches('.');
    if text == "-0" {
        "0".into()
    } else {
        text.into()
    }
}

impl<'a> Figure<'a> {
    fn new(width: f64, height: f64, theme: &'a Theme, title: &str, subtitle: &str) -> Self {
        let mut svg = String::new();
        let _ = write!(
            svg,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif">
<rect width="{width}" height="{height}" fill="{bg}"/>
<text x="28" y="38" font-size="20" font-weight="600" fill="{fg}">{title}</text>
<text x="28" y="60" font-size="13" fill="{muted}">{subtitle}</text>
"#,
            bg = theme.background,
            fg = theme.text,
            muted = theme.muted,
        );
        Self {
            svg,
            theme,
            clips: 0,
        }
    }

    /// Draws a panel with its frame, grid, tick labels and titles, and
    /// returns the axes to plot into. `log_y` labels the vertical axis as
    /// powers of ten (the data passed in must already be `log10`).
    #[allow(clippy::too_many_arguments)]
    fn axes(
        &mut self,
        (left, top, width, height): (f64, f64, f64, f64),
        x: (f64, f64),
        y: (f64, f64),
        title: &str,
        x_label: &str,
        y_label: &str,
        log_y: bool,
    ) -> Axes {
        let t = self.theme;
        self.clips += 1;
        let axes = Axes {
            left,
            top,
            width,
            height,
            x,
            y,
            clip: self.clips,
        };
        let _ = write!(
            self.svg,
            r#"<clipPath id="clip{id}"><rect x="{left}" y="{top}" width="{width}" height="{height}"/></clipPath>
<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="{panel}" stroke="{grid}"/>
<text x="{left}" y="{ty}" font-size="13" font-weight="600" fill="{fg}">{title}</text>
<text x="{cx}" y="{by}" font-size="11" fill="{muted}" text-anchor="middle">{x_label}</text>
<text transform="translate({lx},{cy}) rotate(-90)" font-size="11" fill="{muted}" text-anchor="middle">{y_label}</text>
"#,
            id = axes.clip,
            panel = t.panel,
            grid = t.grid,
            fg = t.text,
            muted = t.muted,
            ty = top - 9.0,
            cx = left + width / 2.0,
            by = top + height + 34.0,
            lx = left - 44.0,
            cy = top + height / 2.0,
        );
        for tick in ticks(x.0, x.1) {
            let (px, _) = axes.to_page((tick, y.0));
            let _ = writeln!(
                self.svg,
                r#"<line x1="{px:.1}" y1="{top}" x2="{px:.1}" y2="{bottom}" stroke="{grid}"/><text x="{px:.1}" y="{ly}" font-size="10" fill="{muted}" text-anchor="middle">{text}</text>"#,
                bottom = top + height,
                ly = top + height + 15.0,
                grid = t.grid,
                muted = t.muted,
                text = label(tick),
            );
        }
        for tick in ticks(y.0, y.1) {
            let (_, py) = axes.to_page((x.0, tick));
            let text = if log_y {
                format!("1e{}", label(tick))
            } else {
                label(tick)
            };
            let _ = writeln!(
                self.svg,
                r#"<line x1="{left}" y1="{py:.1}" x2="{right}" y2="{py:.1}" stroke="{grid}"/><text x="{lx}" y="{ty:.1}" font-size="10" fill="{muted}" text-anchor="end">{text}</text>"#,
                right = left + width,
                lx = left - 6.0,
                ty = py + 3.5,
                grid = t.grid,
                muted = t.muted,
            );
        }
        axes
    }

    fn line(
        &mut self,
        axes: &Axes,
        points: impl IntoIterator<Item = Point>,
        color: &str,
        width: f64,
    ) {
        let mut path = String::new();
        for (i, p) in points.into_iter().enumerate() {
            let (x, y) = axes.to_page(p);
            let _ = write!(path, "{}{x:.2},{y:.2}", if i == 0 { "M" } else { " L" });
        }
        let _ = writeln!(
            self.svg,
            r#"<path d="{path}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" clip-path="url(#clip{id})"/>"#,
            id = axes.clip,
        );
    }

    fn dots(
        &mut self,
        axes: &Axes,
        points: impl IntoIterator<Item = Point>,
        color: &str,
        radius: f64,
    ) {
        let _ = writeln!(
            self.svg,
            r#"<g fill="{color}" clip-path="url(#clip{})">"#,
            axes.clip
        );
        for p in points {
            let (x, y) = axes.to_page(p);
            let _ = writeln!(
                self.svg,
                r#"<circle cx="{x:.2}" cy="{y:.2}" r="{radius}"/>"#
            );
        }
        self.svg.push_str("</g>\n");
    }

    fn circle(&mut self, axes: &Axes, centre: Point, radius: f64, fill: &str, stroke: &str) {
        let (x, y) = axes.to_page(centre);
        let r = radius / (axes.x.1 - axes.x.0) * axes.width;
        let _ = writeln!(
            self.svg,
            r#"<circle cx="{x:.2}" cy="{y:.2}" r="{r:.2}" fill="{fill}" stroke="{stroke}" stroke-dasharray="4 3" clip-path="url(#clip{})"/>"#,
            axes.clip,
        );
    }

    /// An arrow from `from` along `vector`, both in data units.
    fn arrow(&mut self, axes: &Axes, from: Point, vector: Point, color: &str) {
        let (x0, y0) = axes.to_page(from);
        let (x1, y1) = axes.to_page((from.0 + vector.0, from.1 + vector.1));
        let (dx, dy) = (x1 - x0, y1 - y0);
        let length = dx.hypot(dy);
        if length < 1.0 {
            return;
        }
        let (ux, uy) = (dx / length, dy / length);
        let head = 7.0_f64.min(length * 0.5);
        let (bx, by) = (x1 - ux * head, y1 - uy * head);
        let _ = writeln!(
            self.svg,
            r#"<path d="M{x0:.2},{y0:.2} L{x1:.2},{y1:.2} M{x1:.2},{y1:.2} L{ax:.2},{ay:.2} M{x1:.2},{y1:.2} L{cx:.2},{cy:.2}" stroke="{color}" stroke-width="2" stroke-linecap="round" fill="none"/>"#,
            ax = bx - uy * head * 0.5,
            ay = by + ux * head * 0.5,
            cx = bx + uy * head * 0.5,
            cy = by - ux * head * 0.5,
        );
    }

    fn legend(&mut self, axes: &Axes, entries: &[(&str, &str)]) {
        for (i, (text, color)) in entries.iter().enumerate() {
            let y = axes.top + 16.0 + 16.0 * i as f64;
            let x = axes.left + 12.0;
            let _ = writeln!(
                self.svg,
                r#"<line x1="{x}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="3" stroke-linecap="round"/><text x="{tx}" y="{ty}" font-size="11" fill="{fg}">{text}</text>"#,
                x2 = x + 16.0,
                tx = x + 22.0,
                ty = y + 3.5,
                fg = self.theme.text,
            );
        }
    }

    fn save(mut self, directory: &Path, stem: &str) -> Result<PathBuf, Box<dyn Error>> {
        self.svg.push_str("</svg>\n");
        let path = directory.join(format!("{stem}_{}.svg", self.theme.name));
        std::fs::write(&path, self.svg)?;
        Ok(path)
    }
}

/// `(min, max)` of an iterator, padded by a fraction of the span.
fn span(values: impl IntoIterator<Item = f64>, pad: f64) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    let margin = (hi - lo).max(1e-12) * pad;
    (lo - margin, hi + margin)
}

/// `log10` of a relative error, floored so that an exact zero can be drawn.
fn log_error(error: f64) -> f64 {
    error.abs().max(1e-17).log10()
}

// ---------------------------------------------------------------------------
// Figure 1: the double pendulum
// ---------------------------------------------------------------------------

fn double_pendulum(theme: &Theme, out: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let dp = DoublePendulum::default();
    let (q0, dq0) = ([2.2, 2.8], [0.0, 0.0]);
    let tight = SimConfig::dopri5().with_tolerances(1e-10, 1e-12);
    let traj = simulate(&dp, (0.0, 12.0), q0, dq0, &tight)?;

    let mut fig = Figure::new(
        1120.0,
        860.0,
        theme,
        "Double pendulum",
        "Written as one Lagrangian; every curve below is read off the simulated trajectory.",
    );
    let c = theme.series;

    // (a) The motion in space, with the kinematics of both bobs at one instant.
    let a = fig.axes(
        (70.0, 110.0, 440.0, 310.0),
        (-2.9, 2.9),
        (-2.25, 1.85),
        "a  Path of the lower bob, t in [0, 12]",
        "x",
        "y",
        false,
    );
    let path: Vec<Point> = (0..=2400)
        .map(|k| {
            let [_, lower] = traj.kinematics(&dp, 12.0 * k as f64 / 2400.0);
            (lower.pos[0], lower.pos[1])
        })
        .collect();
    fig.line(&a, path, c[0], 1.2);
    let t_shown = 3.36;
    let [upper, lower] = traj.kinematics(&dp, t_shown);
    fig.line(
        &a,
        [
            (0.0, 0.0),
            (upper.pos[0], upper.pos[1]),
            (lower.pos[0], lower.pos[1]),
        ],
        theme.text,
        2.5,
    );
    fig.dots(&a, [(0.0, 0.0)], theme.muted, 3.0);
    fig.dots(
        &a,
        [(upper.pos[0], upper.pos[1]), (lower.pos[0], lower.pos[1])],
        theme.text,
        5.0,
    );
    for bob in [upper, lower] {
        let at = (bob.pos[0], bob.pos[1]);
        fig.arrow(
            &a,
            at,
            (bob.vel[0] * 0.12, bob.vel[1] * 0.12),
            theme.velocity,
        );
        fig.arrow(
            &a,
            at,
            (bob.acc[0] * 0.02, bob.acc[1] * 0.02),
            theme.acceleration,
        );
    }
    fig.legend(
        &a,
        &[
            ("velocity", theme.velocity),
            ("acceleration", theme.acceleration),
        ],
    );

    // (b) Phase portrait of the second coordinate.
    let phase: Vec<Point> = (0..=6000)
        .map(|k| {
            let (q, p) = traj.phase_point(&dp, 12.0 * k as f64 / 6000.0);
            (q[1], p[1])
        })
        .collect();
    let b = fig.axes(
        (610.0, 110.0, 440.0, 310.0),
        span(phase.iter().map(|p| p.0), 0.05),
        span(phase.iter().map(|p| p.1), 0.05),
        "b  Phase portrait: angle and momentum of the lower rod",
        "theta2 (rad)",
        "p2",
        false,
    );
    fig.line(&b, phase, c[1], 1.0);

    // (c) Poincaré sections at three energies, from regular to chaotic.
    let mut sections = Vec::new();
    for (start, color) in [([0.4, 0.5], c[3]), ([1.0, 1.4], c[0]), ([1.6, 2.2], c[4])] {
        let long = simulate(&dp, (0.0, 600.0), start, [0.0, 0.0], &SimConfig::default())?;
        let points: Vec<Point> = long
            .crossings(|s| s.q[0], CrossingDirection::Rising)
            .iter()
            .map(|s| {
                let wrapped = (s.q[1] + PI).rem_euclid(2.0 * PI) - PI;
                (wrapped, momentum(&dp, s.q, s.dq, s.t)[1])
            })
            .collect();
        sections.push((points, color));
    }
    let all = sections.iter().flat_map(|(p, _)| p.iter().copied());
    let cx = fig.axes(
        (70.0, 505.0, 440.0, 290.0),
        span(all.clone().map(|p| p.0), 0.05),
        span(all.map(|p| p.1), 0.05),
        "c  Poincaré section theta1 = 0 (rising), three energies, t in [0, 600]",
        "theta2 (rad)",
        "p2",
        false,
    );
    for (points, color) in sections {
        fig.dots(&cx, points, color, 1.6);
    }

    // (d) Long-run energy error: adaptive against symplectic.
    let t_end = 400.0;
    let loose = SimConfig::dopri5().with_tolerances(1e-6, 1e-8);
    let adaptive = simulate(&dp, (0.0, t_end), q0, dq0, &loose)?;
    let fixed = SimConfig::variational4(0.005);
    let symplectic = simulate(&dp, (0.0, t_end), q0, dq0, &fixed)?;
    let error_curve = |traj: &Trajectory<2>| -> Vec<Point> {
        let report = traj.energy_drift_at_knots(&dp);
        // The running maximum: an envelope is easier to read than a band.
        let mut worst = 1e-17_f64;
        let stride = (traj.len_segments() / 1500).max(1);
        traj.knots()
            .enumerate()
            .filter_map(|(i, s)| {
                let e = energy(&dp, s.q, s.dq, s.t).total;
                worst = worst.max((e - report.initial).abs() / report.scale);
                (i % stride == 0).then(|| (s.t, log_error(worst)))
            })
            .collect()
    };
    let (ea, es) = (error_curve(&adaptive), error_curve(&symplectic));
    let d = fig.axes(
        (610.0, 505.0, 440.0, 290.0),
        (0.0, t_end),
        (-8.0, -2.0),
        "d  Largest relative energy error so far",
        "t",
        "max |E - E0| / scale",
        true,
    );
    fig.line(&d, ea, c[1], 1.8);
    fig.line(&d, es, c[0], 1.8);
    fig.legend(
        &d,
        &[
            ("Dopri5, rtol 1e-6: keeps growing", c[1]),
            ("Variational4, dt 0.005: stays in a band", c[0]),
        ],
    );

    fig.save(out, "double_pendulum")
}

// ---------------------------------------------------------------------------
// Figure 2: the bead on a rotating hoop
// ---------------------------------------------------------------------------

fn rotating_hoop(theme: &Theme, out: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let hoop = BeadOnRotatingHoop::default();
    let equilibrium = hoop
        .equilibrium_angle()
        .ok_or("the hoop spins too slowly")?;
    let c = theme.series;
    let mut fig = Figure::new(
        1120.0,
        470.0,
        theme,
        "Bead on a rotating hoop",
        "Above the critical spin rate the bottom is unstable and two tilted equilibria appear at cos(theta) = g / (R Omega^2).",
    );

    // (a) The effective potential the engine reports, V_eff = -L(q, 0, t).
    let potential: Vec<Point> = (-300..=300)
        .map(|k| {
            let theta = k as f64 / 300.0 * 2.4;
            (theta, energy(&hoop, [theta], [0.0], 0.0).potential)
        })
        .collect();
    let a = fig.axes(
        (70.0, 110.0, 290.0, 290.0),
        (-2.4, 2.4),
        span(potential.iter().map(|p| p.1), 0.08),
        "a  Effective potential",
        "theta (rad)",
        "V_eff",
        false,
    );
    fig.line(&a, potential, c[0], 2.0);
    let well = energy(&hoop, [equilibrium], [0.0], 0.0).potential;
    fig.dots(&a, [(equilibrium, well), (-equilibrium, well)], c[3], 4.5);
    fig.dots(
        &a,
        [(0.0, energy(&hoop, [0.0], [0.0], 0.0).potential)],
        c[4],
        4.5,
    );

    // (b) Phase portrait: orbits inside each well, and around both.
    let b = fig.axes(
        (440.0, 110.0, 290.0, 290.0),
        (-2.4, 2.4),
        (-9.5, 9.5),
        "b  Phase portrait",
        "theta (rad)",
        "d(theta)/dt",
        false,
    );
    let tight = SimConfig::dopri5().with_tolerances(1e-10, 1e-12);
    for (start, color) in [
        (0.9, c[3]),
        (0.5, c[3]),
        (0.05, c[4]),
        (-0.9, c[1]),
        (-0.5, c[1]),
        (2.0, c[2]),
    ] {
        let traj = simulate(&hoop, (0.0, 12.0), [start], [0.0], &tight)?;
        let orbit: Vec<Point> = (0..=3000)
            .map(|k| {
                let s = traj.sample(12.0 * k as f64 / 3000.0);
                (s.q[0], s.dq[0])
            })
            .collect();
        fig.line(&b, orbit, color, 1.3);
    }
    fig.dots(&b, [(equilibrium, 0.0), (-equilibrium, 0.0)], c[3], 4.0);

    // (c) Which energy is conserved.
    let traj = simulate(&hoop, (0.0, 12.0), [0.3], [0.0], &tight)?;
    let report = traj.energy_drift_at_knots(&hoop);
    let naive = |s: &State<1>| {
        // ½ q̇ᵀ M q̇ + V with V = −m g R cos θ: what a formula that assumes
        // a purely quadratic kinetic energy would report.
        let m = mass_matrix(&hoop, s.q, s.dq, s.t)[0][0];
        0.5 * m * s.dq[0] * s.dq[0] - hoop.m * hoop.g * hoop.radius * s.q[0].cos()
    };
    let first_naive = naive(&traj.start());
    let (mut jacobi, mut wrong) = (Vec::new(), Vec::new());
    for s in traj.knots() {
        let e = energy(&hoop, s.q, s.dq, s.t).total;
        jacobi.push((s.t, log_error((e - report.initial) / report.scale)));
        wrong.push((s.t, log_error((naive(&s) - first_naive) / report.scale)));
    }
    // Room above the data for the legend.
    let d = fig.axes(
        (810.0, 110.0, 260.0, 290.0),
        (0.0, 12.0),
        (-16.0, 5.0),
        "c  Which energy is conserved",
        "t",
        "|E - E0| / scale",
        true,
    );
    fig.line(&d, wrong, c[1], 1.5);
    fig.line(&d, jacobi, c[0], 1.5);
    fig.legend(
        &d,
        &[
            ("quadratic T + V: not conserved", c[1]),
            ("Jacobi integral p.q' - L", c[0]),
        ],
    );

    fig.save(out, "rotating_hoop")
}

// ---------------------------------------------------------------------------
// Figure 3: geodesics around a black hole
// ---------------------------------------------------------------------------

fn schwarzschild(theme: &Theme, out: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let hole = SchwarzschildGeodesic { rs: 1.0 };
    let tight = SimConfig::dopri5().with_tolerances(1e-11, 1e-13);
    let c = theme.series;
    let mut fig = Figure::new(
        1120.0,
        470.0,
        theme,
        "Geodesics of a Schwarzschild black hole",
        "L is half the squared length of the four-velocity; its mass matrix is the metric, with signature (-, +, +, +).",
    );
    let in_plane = |s: &State<4>| (s.q[1] * s.q[3].cos(), s.q[1] * s.q[3].sin());
    let hole_marks = |fig: &mut Figure<'_>, axes: &Axes| {
        fig.circle(axes, (0.0, 0.0), hole.photon_sphere(), "none", theme.muted);
        fig.circle(axes, (0.0, 0.0), hole.rs, theme.text, "none");
    };

    // (a) A bound orbit: it precesses by almost half a turn per revolution.
    let q0 = [0.0, 10.0, FRAC_PI_2, 0.0];
    let u0 = hole.timelike_velocity(10.0, 0.0, 0.02);
    let orbit = simulate(&hole, (0.0, 1500.0), q0, u0, &tight)?;
    let a = fig.axes(
        (70.0, 110.0, 290.0, 290.0),
        (-11.0, 11.0),
        (-11.0, 11.0),
        "a  A massive particle: precessing orbit",
        "x / r_s",
        "y / r_s",
        false,
    );
    fig.line(
        &a,
        (0..=6000).map(|k| in_plane(&orbit.sample(1500.0 * k as f64 / 6000.0))),
        c[0],
        1.1,
    );
    hole_marks(&mut fig, &a);

    // (b) Light rays arriving from the right with different impact
    // parameters: bent, whirled around the photon sphere, or captured.
    let b = fig.axes(
        (440.0, 110.0, 290.0, 290.0),
        (-9.0, 9.0),
        (-9.0, 9.0),
        "b  Light: deflected, whirling, captured",
        "x / r_s",
        "y / r_s",
        false,
    );
    for (impact, color) in [
        (6.0, c[3]),
        (4.0, c[3]),
        (3.0, c[0]),
        (2.65, c[2]),
        (2.55, c[1]),
        (2.0, c[1]),
        (1.0, c[1]),
    ] {
        // Start far to the right, moving left, offset by the impact parameter.
        let (x, y) = (30.0_f64, impact);
        let (r, phi) = (x.hypot(y), y.atan2(x));
        // A unit Cartesian velocity (-1, 0) in polar components.
        let (dr, dphi) = (-phi.cos(), phi.sin() / r);
        let start = [0.0, r, FRAC_PI_2, phi];
        let mut stepper = Stepper::new(&hole, 0.0, start, hole.null_velocity(r, dr, dphi), tight)?;
        let mut ray = vec![in_plane(&stepper.state())];
        while stepper.state().t < 90.0 {
            let Ok(segment) = stepper.step_bounded(90.0) else {
                break; // reached the horizon, where these coordinates end
            };
            ray.extend((1..=8).map(|k| in_plane(&segment.eval(k as f64 / 8.0))));
            if stepper.state().q[1] < 1.02 * hole.rs {
                break;
            }
        }
        fig.line(&b, ray, color, 1.3);
    }
    hole_marks(&mut fig, &b);

    // (c) The constants of the motion along the orbit of panel a.
    let p0 = momentum(&hole, q0, u0, 0.0);
    let norm0 = hole.lagrangian(q0, u0, 0.0);
    let mut curves = [Vec::new(), Vec::new(), Vec::new()];
    for s in orbit.knots().step_by(4) {
        let p = momentum(&hole, s.q, s.dq, s.t);
        curves[0].push((s.t, log_error((p[0] - p0[0]) / p0[0])));
        curves[1].push((s.t, log_error((p[3] - p0[3]) / p0[3])));
        curves[2].push((
            s.t,
            log_error((hole.lagrangian(s.q, s.dq, s.t) - norm0) / norm0),
        ));
    }
    let d = fig.axes(
        (810.0, 110.0, 260.0, 290.0),
        (0.0, 1500.0),
        (-17.0, -8.0),
        "c  Constants of the motion",
        "proper time",
        "relative drift",
        true,
    );
    for (curve, color) in curves.into_iter().zip([c[0], c[1], c[3]]) {
        fig.line(&d, curve, color, 1.2);
    }
    fig.legend(
        &d,
        &[
            ("energy p_t", c[0]),
            ("angular momentum p_phi", c[1]),
            ("norm of the four-velocity", c[3]),
        ],
    );

    fig.save(out, "schwarzschild")
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "target/svg_showcase".into()),
    );
    std::fs::create_dir_all(&out)?;
    for theme in [&DARK, &LIGHT] {
        for path in [
            double_pendulum(theme, &out)?,
            rotating_hoop(theme, &out)?,
            schwarzschild(theme, &out)?,
        ] {
            println!("wrote {}", path.display());
        }
    }
    Ok(())
}
