use crate::lalg::{matrix::Matrix, vector::Vector};

use super::{intersection::Intersection, ray::{Intersects, Ray}};

#[derive(Debug, Clone, PartialEq)]
pub struct Sphere {
  pub origin: Vector<4>,
  pub radius: f64,
  pub transform: Matrix<4, 4>
}

impl Sphere {
  pub fn new() -> Self { 
    Self{ origin: Vector::with_dimension(), radius: 0.0, transform: Matrix::identity().unwrap() }
  }

  pub fn with(origin: Vector<4>, radius: f64) -> Self {
    Self{ origin, radius, transform: Matrix::identity().unwrap() }
  }

  pub fn normal_at(&self, point: &Vector<4>) -> Vector<4> {
    let object_point = &(self.transform.inverse()).unwrap() * &Matrix::from(point.clone());
    let object_normal = &object_point - &Matrix::from(self.origin.clone());
    let world_point: Matrix<4, 1> = &(&(self.transform.inverse()).unwrap()).transpose() * &object_normal;
    let mut result = Vector::from(world_point.transpose::<1,4>()[0].clone());

    result[3] = 0.0;

    result.norm()
    // (point - &self.origin).norm()
  }
}

impl Intersects<Self> for Sphere {
  fn intersections<'a>(&'a self, ray: &Ray) -> Vec<Intersection<'a, Sphere>> {
    let transformed_ray = ray.transform(&self.transform);
    let sphere_to_ray = &transformed_ray.origin - &self.origin;
    let a = transformed_ray.direction.dot(&transformed_ray.direction);
    let b = 2.0 * transformed_ray.direction.dot(&sphere_to_ray);
    let c = sphere_to_ray.dot(&sphere_to_ray) - 1.0;
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
      origin: Vector::from([0.0, 0.0, 0.0, 0.0]),
      radius: 1.0,
      transform: Matrix::identity().unwrap()
    };
    let ray = Ray{
      origin: Vector::from([0.0, 0.0, -5.0, 0.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = [
      Intersection{ t: 4.0, object: &sphere },
      Intersection{ t: 6.0, object: &sphere },
    ];

    // intersects on tangent
    assert_eq!(expected.len(), intersections.len());
    for i in 0..expected.len() {
      assert_eq!(expected[i].t, intersections[i].t);
    }
    // assert_eq!(intersections, expected);

    let ray = Ray{
      origin: Vector::from([0.0, 1.0, -5.0, 0.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = [
      Intersection{ t: 5.0, object: &sphere },
      Intersection{ t: 5.0, object: &sphere },
    ];

    assert_eq!(intersections, expected);

    // ray misses
    let ray = Ray{
      origin: Vector::from([0.0, 2.0, -5.0, 0.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    let expected = vec![];

    assert_eq!(intersections, expected);
  }
}