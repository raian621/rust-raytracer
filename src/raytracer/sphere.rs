use crate::lalg::vector::Vector;

use super::{intersection::Intersection, ray::{Intersects, Ray}};

#[derive(Debug, Clone, PartialEq)]
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

impl Intersects<Self> for Sphere {
  fn intersections<'a>(&'a self, ray: &Ray) -> Vec<Intersection<'a, Sphere>> {
    let sphere_to_ray = (&ray.origin - &self.origin).unwrap();
    let a = ray.direction.dot(&ray.direction).unwrap();
    let b = 2.0 * ray.direction.dot(&sphere_to_ray).unwrap();
    let c = sphere_to_ray.dot(&sphere_to_ray).unwrap() - 1.0;
    let discriminant = b.powi(2) - 4.0 * a * c;

    if discriminant < 0.0 {
      return Vec::<Intersection<'a, Sphere>>::with_capacity(0);
    }

    vec![
      Intersection{t: (-b - discriminant.sqrt()) / (2.0*a), object: self},
      Intersection{t: (-b + discriminant.sqrt()) / (2.0*a), object: self}
    ]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sphere_intersects() {
    // two points through center
    let sphere = Sphere{
      origin: Vector::from(vec![0.0, 0.0, 0.0, 0.0]),
      radius: 1.0
    };
    let ray = Ray{
      origin: Vector::from(vec![0.0, 0.0, -5.0, 0.0]),
      direction: Vector::from(vec![0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = vec![
      Intersection{ t: 4.0, object: &sphere },
      Intersection{ t: 6.0, object: &sphere },
    ];

    // intersects on tangent
    assert_eq!(intersections, expected);

    let ray = Ray{
      origin: Vector::from(vec![0.0, 1.0, -5.0, 0.0]),
      direction: Vector::from(vec![0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = vec![
      Intersection{ t: 5.0, object: &sphere },
      Intersection{ t: 5.0, object: &sphere },
    ];

    assert_eq!(intersections, expected);

    // ray misses
    let ray = Ray{
      origin: Vector::from(vec![0.0, 2.0, -5.0, 0.0]),
      direction: Vector::from(vec![0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = vec![];

    assert_eq!(intersections, expected);
  }
}