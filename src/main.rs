use std::ops::{Add, Sub, Mul};
use std::vec;
use std::cmp::Ordering::{Less, Equal, Greater};
use std::f64::consts::PI;
use std::fs::File;
use std::io::prelude::*;

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
struct Hit(f64, V3, V3, Material);

trait Solid {
    fn intersect(&self, r: &Ray) -> Option<Hit>;
}

fn min_positive(hits: Vec<Option<Hit>>) -> Option<Hit> {
    let mut sorted = hits.iter().filter(|x| {
        if let Some(x) = x {
            x.0 > 0.0
        } else {
            false
        }
    }).collect::<Vec<&Option<Hit>>>();
    sorted.sort_by(|a, b| {
        if let (Some(a), Some(b)) = (a, b) {
            if a.0 < b.0 {
                Less
            } else if a.0 > b.0 {
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
            Some(Hit(disp.magnitude(), hit, self.normal, self.mat))
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
                Some(Hit(disp.magnitude(), hit, normal, self.mat))
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
            Some(Hit(dispx1.magnitude(), r.point + dispx1, V3(-1.0, 0.0, 0.0), self.mat)),
            Some(Hit(dispx2.magnitude(), r.point + dispx2, V3( 1.0, 0.0, 0.0), self.mat)),
            Some(Hit(dispy1.magnitude(), r.point + dispy1, V3(0.0, -1.0, 0.0), self.mat)),
            Some(Hit(dispy2.magnitude(), r.point + dispy2, V3(0.0,  1.0, 0.0), self.mat)),
            Some(Hit(dispz1.magnitude(), r.point + dispx1, V3(0.0, 0.0, -1.0), self.mat)),
            Some(Hit(dispz2.magnitude(), r.point + dispx2, V3(0.0, 0.0,  1.0), self.mat)),
        ];
        min_positive(hits)
    }
}

struct Light {
    point: V3,
    color: Color,
}

impl Light {
    fn shade(&self, h: &Hit) -> Color {
        let mut light_vec = self.point - h.1;
        light_vec.normalize();
        let intensity = light_vec * h.2;
        let (cl, cm) = (self.color, h.3.color);
        Color(
            ((cl.0 as f64 / 255.0) * (cm.0 as f64 / 255.0) * intensity * 255.0) as u8,
            ((cl.1 as f64 / 255.0) * (cm.1 as f64 / 255.0) * intensity * 255.0) as u8,
            ((cl.2 as f64 / 255.0) * (cm.2 as f64 / 255.0) * intensity * 255.0) as u8,
        )
    }
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
    solids: Vec<Box<dyn Solid>>,
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
    fn shade(&self, h: &Hit) -> Color {
        let mc = h.3.color;
        let mut shaded_c = Color(0, 0, 0);
        for l in &self.lights {
            let light_vec = l.point - h.1;
            light_vec.normalize();
            let intensity = light_vec * h.2;
            let lc = l.color;
            let l_shaded_color = (lc * mc).scale(intensity);
            shaded_c = shaded_c + l_shaded_color;
        }
        shaded_c
    }
    fn render(&self, path: &str) -> std::io::Result<()> {
        let rays = self.camera.rays();
        let mut hits = Vec::<Option<Hit>>::new();
        for ray in &rays {
            hits.push(self.intersect(ray));
        }
        let mut image = File::create(path)?;
        write!(image, "P3\n")?;
        write!(image, "{} {}\n", self.camera.resolution.0, self.camera.resolution.1)?;
        write!(image, "255\n")?;
        let bg = self.background;
        for (i, h) in hits.iter().enumerate() {
            if let Some(h) = h {
                let color = self.shade(h);
                write!(image, "{} {} {} ", color.0, color.1, color.2)?;
            } else {
                write!(image, "{} {} {} ", bg.0, bg.1, bg.2)?;
            }
            if (i as i64 + 1) % self.camera.resolution.0 == 0 {
                write!(image, "\n")?;
            }
        }
        Ok(())
    }
}

fn main() {
    let scene = Scene {
        solids: vec![
            Box::new(Plane{
                point: V3(0.0, 0.0, 0.0),
                normal: V3(0.0, 1.0, 0.0),
                mat: Material {
                    color: Color(0, 0, 255),
                    reflection: 0.2,
                },
            }),
            Box::new(Sphere{
                point: V3(0.0, 1.0, 0.0),
                radius: 1.0,
                mat: Material {
                    color: Color(0, 255, 0),
                    reflection: 0.0,
                },
            }),
            Box::new(Sphere{
                point: V3(-2.0, 1.0, 0.0),
                radius: 1.0,
                mat: Material {
                    color: Color(255, 0, 0),
                    reflection: 0.0,
                },
            }),
            Box::new(Sphere{
                point: V3(-4.0, 1.0, 0.0),
                radius: 1.0,
                mat: Material {
                    color: Color(0, 255, 0),
                    reflection: 0.0,
                },
            }),
            Box::new(Sphere{
                point: V3(2.0, 1.0, 0.0),
                radius: 1.0,
                mat: Material {
                    color: Color(255, 0, 0),
                    reflection: 0.0,
                },
            }),
            Box::new(Sphere{
                point: V3(4.0, 1.0, 0.0),
                radius: 1.0,
                mat: Material {
                    color: Color(0, 255, 0),
                    reflection: 0.0,
                },
            }),
        ],
        lights: vec![
            Light{point: V3(-2.5, 5.0, -2.5), color: Color(80,  0, 80)},
            Light{point: V3( 2.5, 5.0, -2.5), color: Color( 0, 80, 80)},
        ],
        camera: Camera {
            point: V3(0.0, 1.0, -5.0),
            direction: V3(0.0, 0.0, 1.0),
            fov: PI/2.0,
            resolution: (1920, 1080),
        },
        background: Color(212, 212, 212),
    };
    if let Err(e) = scene.render("out.ppm") {
        println!("{e}");
    }
}

