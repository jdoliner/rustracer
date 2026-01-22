use std::ops::{Add, Sub, Mul};
use std::vec;
use std::cmp::Ordering::{Less, Equal, Greater};
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;
use rayon::prelude::*;
use indicatif::{ProgressIterator, ParallelProgressIterator, ProgressBar, ProgressStyle};

const EPSILON: f64 = 1e-9;
#[derive(Debug,Copy,Clone)]
struct V3 (f64, f64, f64);

impl Add for V3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        V3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl Sub for V3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        V3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl Mul for V3 {
    type Output = f64;

    fn mul(self, other: Self) -> f64 {
        (self.0 * other.0) + (self.1 * other.1) + (self.2 * other.2)
    }
}

impl V3 {
    fn magnitude(self) -> f64 {
        (self*self).sqrt()
    }
    fn scale(self, s: f64) -> V3 {
        V3(self.0 * s, self.1 * s, self.2 * s)
    }
    fn normalize(mut self) {
        self = self.scale(self.magnitude());
    }
    fn cross(self, other: V3) -> V3 {
        V3((self.1*other.2) - (self.2*other.1), (self.0*other.2) - (self.2*other.0), (self.0*other.1) - (self.1*other.0))
    }
}

struct Ray {
    point: V3,
    direction: V3,
}

#[derive(Debug,Copy,Clone)]
struct Color (u8, u8, u8);

impl Add for Color {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Color(
            (self.0 as u16 + other.0 as u16).clamp(0, 255) as u8,
            (self.1 as u16 + other.1 as u16).clamp(0, 255) as u8,
            (self.2 as u16 + other.2 as u16).clamp(0, 255) as u8,
        )
    }
}

impl Mul for Color {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Color(
            ((self.0 as f64 / 255.0) * (other.0 as f64 / 255.0) * 255.0) as u8, 
            ((self.1 as f64 / 255.0) * (other.1 as f64 / 255.0) * 255.0) as u8,
            ((self.2 as f64 / 255.0) * (other.2 as f64 / 255.0) * 255.0) as u8,
        )
    }
}

impl Color {
    fn scale(self, s: f64) -> Self {
        Color(
            (self.0 as f64 * s).clamp(0.0, 255.0) as u8,
            (self.1 as f64 * s).clamp(0.0, 255.0) as u8,
            (self.2 as f64 * s).clamp(0.0, 255.0) as u8,
        )
    }
}

#[derive(Debug,Copy,Clone)]
struct Material {
    color: Color,
    reflection: f64,
}

#[derive(Debug,Copy,Clone)]
struct Hit {
    distance: f64,
    direction: V3,
    point: V3,
    normal: V3,
    mat: Material,
}

trait Solid {
    fn intersect(&self, r: &Ray) -> Option<Hit>;
}

fn min_positive(hits: Vec<Option<Hit>>) -> Option<Hit> {
    let mut sorted = hits.iter().filter(|x| {
        if let Some(x) = x {
            x.distance > 0.0
        } else {
            false
        }
    }).collect::<Vec<&Option<Hit>>>();
    sorted.sort_by(|a, b| {
        if let (Some(a), Some(b)) = (a, b) {
            if a.distance < b.distance {
                Less
            } else if a.distance > b.distance {
                Greater
            } else {
                Equal
            }
        } else {
            panic!("unreachable")
        }
    });
    if sorted.is_empty() {
        None
    } else {
        *sorted[0]
    }
}

struct Plane {
    point: V3,
    normal: V3,
    mat: Material,
}

impl Solid for Plane {
    fn intersect(&self, r: &Ray) -> Option<Hit> {
        let denom = r.direction * self.normal;
        if denom.abs() < EPSILON {
            return None
        }
        let t = ((self.point - r.point) * self.normal) / denom;
        if t < 0.0 {
            None
        } else {
            let disp = r.direction.scale(t);
            let hit = r.point + disp;
            Some(Hit{
                distance: disp.magnitude(),
                direction: r.direction,
                point: hit,
                normal: self.normal,
                mat: self.mat,
            })
        }
    }
}

struct Sphere {
    point: V3,
    radius: f64,
    mat: Material,
}

impl Solid for Sphere {
    fn intersect(&self, r: &Ray) -> Option<Hit> {
        let L = r.point - self.point;
        // we have a quadratic equation at^2 + bt + c = 0 where:
        let a = r.direction * r.direction;
        let b = 2.0 * (r.direction * L);
        let c = (L * L) - (self.radius * self.radius);

        let discriminant = (b*b) - (4.0*a*c);
        if discriminant < 0.0 {
            None
        } else {
            let discriminant = discriminant.sqrt();
            let t1 = (-b + discriminant) / (2.0 * a);
            let t2 = (-b - discriminant) / (2.0 * a);
            let t = if t1 < 0.0 && t2 < 0.0 {
                None
            } else if t1 < 0.0 {
                Some(t2)
            } else if t2 < 0.0 {
                Some(t1)
            } else if t1 < t2 {
                Some(t1)
            } else {
                Some(t2)
            };
            if let Some(t) = t {
                let disp = r.direction.scale(t);
                let hit = r.point + disp;
                let normal = (hit - self.point).scale(1.0/self.radius);
                Some(Hit{
                    distance: disp.magnitude(),
                    direction: r.direction,
                    point: hit,
                    normal: normal,
                    mat: self.mat,
                })
            } else {
                None
            }
        }
    }
}

struct Cube {
    point: V3,
    side: f64,
    mat: Material,
}

impl Solid for Cube {
    fn intersect(&self, r: &Ray) -> Option<Hit> {
        let radius = self.side / 2.0;
        let (x_min, x_max) = (self.point.0 - radius, self.point.0 + radius);
        let (y_min, y_max) = (self.point.1 - radius, self.point.1 + radius);
        let (z_min, z_max) = (self.point.2 - radius, self.point.2 + radius);

        let (tx1, tx2) = ((x_min - r.point.0) / r.direction.0, (x_max - r.point.0) / r.direction.0);
        let (ty1, ty2) = ((y_min - r.point.0) / r.direction.0, (y_max - r.point.0) / r.direction.0);
        let (tz1, tz2) = ((z_min - r.point.0) / r.direction.0, (z_max - r.point.0) / r.direction.0);

        let (dispx1, dispx2) = (r.direction.scale(tx1), r.direction.scale(tx2));
        let (dispy1, dispy2) = (r.direction.scale(ty1), r.direction.scale(ty2));
        let (dispz1, dispz2) = (r.direction.scale(tz1), r.direction.scale(tz2));

        let hits = vec![
            Some(Hit{
                distance: dispx1.magnitude(),
                direction: r.direction,
                point: r.point + dispx1,
                normal: V3(-1.0, 0.0, 0.0),
                mat: self.mat,
            }),
            Some(Hit{
                distance: dispx2.magnitude(),
                direction: r.direction,
                point: r.point + dispx2,
                normal: V3( 1.0, 0.0, 0.0),
                mat: self.mat,
            }),
            Some(Hit{
                distance: dispy1.magnitude(),
                direction: r.direction,
                point: r.point + dispy1,
                normal: V3(0.0, -1.0, 0.0),
                mat: self.mat,
            }),
            Some(Hit{
                distance: dispy2.magnitude(),
                direction: r.direction,
                point: r.point + dispy2,
                normal: V3(0.0,  1.0, 0.0),
                mat: self.mat,
            }),
            Some(Hit{
                distance: dispz1.magnitude(),
                direction: r.direction,
                point: r.point + dispx1,
                normal: V3(0.0, 0.0, -1.0),
                mat: self.mat,
            }),
            Some(Hit{
                distance: dispz2.magnitude(),
                direction: r.direction,
                point: r.point + dispx2,
                normal: V3(0.0, 0.0,  1.0),
                mat: self.mat,
            }),
        ];
        min_positive(hits)
    }
}

struct Light {
    point: V3,
    color: Color,
}

struct Camera {
    point: V3,
    direction: V3,
    fov: f64,
    resolution: (i64, i64),
}

impl Camera {
    fn rays(&self) -> Vec<Ray> {
        let mut result = Vec::<Ray>::new();
        let aspect_ratio = self.resolution.0 as f64 / self.resolution.1 as f64;
        let width = 2.0 * (self.fov / 2.0).tan();
        let height = width / aspect_ratio;
        let W = self.direction;
        let U = V3(0.0, 1.0, 0.0).cross(W);
        let V = W.cross(U);
        for j in (0..self.resolution.1).rev() {
            for i in 0..self.resolution.0 {
                let (u, v) = ((i as f64 + 0.5)/self.resolution.0 as f64, (j as f64 + 0.5)/self.resolution.1 as f64);
                let (x, y) = (((2.0 * u) - 1.0) * (width / 2.0), (1.0 - (2.0 * v)) * (height / 2.0));
                let direction = U.scale(x) + V.scale(y) + W;
                direction.normalize();
                result.push(Ray {point: self.point, direction});
            }
        }
        result
    }
}

struct Scene {
    solids: Vec<Box<dyn Solid+Send+Sync>>,
    lights: Vec<Light>,
    camera: Camera,
    background: Color,
}

impl Solid for Scene {
    fn intersect(&self, r: &Ray) -> Option<Hit> {
        let mut hits = Vec::<Option<Hit>>::new();
        for s in &self.solids {
            hits.push(s.intersect(r));
        }
        min_positive(hits)
    }
}

impl Scene {
    fn shade(&self, h: &Hit, bounces: i64) -> Color {
        let mc = h.mat.color;
        let mut shaded_c = Color(0, 0, 0);
        for l in &self.lights {
            let light_dir = l.point - h.point;
            let light_dist = light_dir.magnitude();
            light_dir.normalize();
            let sh = &self.intersect(&Ray{
                point: h.point + h.normal.scale(0.0001),
                direction: light_dir,
            });
            if let Some(sh) = sh && sh.distance < light_dist {
                continue
            }
            let intensity = light_dir * h.normal;
            let lc = l.color;
            let l_shaded_color = (lc * mc).scale(intensity);
            shaded_c = shaded_c + l_shaded_color;
        }
        let r = h.mat.reflection;
        let mut reflection_c = self.background;
        if r > EPSILON && bounces > 0 {
            let r_direction = h.direction - h.normal.scale(2.0 * (h.direction * h.normal));
            let r_point = h.point + h.normal.scale(0.0001);
            let rh = &self.intersect(&Ray{point: r_point, direction: r_direction});
            if let Some(rh) = rh {
                reflection_c = self.shade(rh, bounces - 1);
            }
        }
        shaded_c.scale(1.0 - r) + reflection_c.scale(r)
    }
    fn render(&self, path: &str) -> std::io::Result<()> {
        let rays = self.camera.rays();
        let bg = self.background;
        let pb = ProgressBar::new(rays.len() as u64);
        pb.set_style(ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) | {msg} | Speed: {per_sec} | ETA: {eta}"
    )
            .unwrap()
            .progress_chars("=> "));
        let colors: Vec::<Color> = rays.par_iter()
            .progress_with(pb.clone())
            .map(|r| {
                let h = &self.intersect(r);
                if let Some(h) = h {
                    self.shade(h, 4)
                } else {
                    bg
                }
            })
            .collect();
        let mut image = File::create(path)?;
        write!(image, "P3\n")?;
        write!(image, "{} {}\n", self.camera.resolution.0, self.camera.resolution.1)?;
        write!(image, "255\n")?;
        pb.reset();
        for (i, c) in colors.iter().progress_with(pb.clone()).enumerate() {
            write!(image, "{} {} {} ", c.0, c.1, c.2)?;
            if (i as i64 + 1) % self.camera.resolution.0 == 0 {
                write!(image, "\n")?;
            }
        }
        Ok(())
    }
}

fn main() {
    let mut scene = Scene {
        solids: vec![
            Box::new(Plane{
                point: V3(0.0, 0.0, 0.0),
                normal: V3(0.0, 1.0, 0.0),
                mat: Material {
                    color: Color(0, 0, 255),
                    reflection: 0.4,
                },
            }),
        ],
        lights: vec![
            Light{point: V3(-2.5, 10.0, -2.5), color: Color(80,  0, 80)},
            Light{point: V3( 2.5, 10.0, -2.5), color: Color( 0, 80, 80)},
        ],
        camera: Camera {
            point: V3(0.0, 5.0, -10.0),
            direction: V3(0.0, 0.0, 1.0),
            fov: PI/2.0,
            resolution: (1920, 1080),
        },
        background: Color(212, 212, 212),
    };
    for i in -3..3 {
        for j in 1..7 {
            for k in 0..6{
                scene.solids.push(
                    Box::new(Sphere{
                        point: V3(i as f64, j as f64 - 0.5, k as f64),
                        radius: 0.5,
                        mat: Material {
                            color: Color(255, 255, 255),
                            reflection: 0.2,
                        },
                    }));
            }
        }
    }
    if let Err(e) = scene.render("out.ppm") {
        println!("{e}");
    }
}

