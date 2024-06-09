use crate::lalg::vector::Vector;

use super::ray::{Ray, Intersects};

pub struct Sphere {
  pub origin: Vector<f64>,
  pub radius: f64
}

impl Sphere {
  pub fn new() -> Self { Self{ origin: Vector::with_dimension(4), radius: 0.0 }}
  pub fn with(origin: Vector<f64>, radius: f64) -> Self {
    Self{ origin, radius }
  }
}

impl Intersects for Sphere {
  fn intersections(&self, ray: Ray) -> Vec<f64> {
    let sphere_to_ray = (&ray.origin - &self.origin).unwrap();
    let a = ray.direction.dot(&ray.direction).unwrap();
    let b = 2.0 * ray.direction.dot(&sphere_to_ray).unwrap();
    let c = sphere_to_ray.dot(&sphere_to_ray).unwrap() - 1.0;
    let discriminant = b.powi(2) - 4.0 * a * c;

    match discriminant as i64 {
      0 => Vec::with_capacity(0),
      _ => vec![-b + discriminant.sqrt() / 2.0*a, -b - discriminant.sqrt() / 2.0*a]
    }
  }
}