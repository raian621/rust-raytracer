use crate::{lalg::{matrix::Matrix, vector::Vector}, raytracer::ray::Ray};

use super::{intersection::{Intersection, Intersects}, material::Material};

#[derive(Debug, Clone, PartialEq)]
pub struct Sphere {
  pub origin: Vector<4>,
  pub radius: f64,
  pub transform: Matrix<4, 4>,
  pub material: Material
}

impl Sphere {
  pub fn new() -> Self { 
    Self{
      origin: Vector::from([0.0, 0.0, 0.0, 1.0]),
      radius: 1.0,
      transform: Matrix::identity().unwrap(),
      material: Material::default()
    }
  }

  pub fn with(origin: Vector<4>, radius: f64) -> Self {
    Self{ origin, radius, transform: Matrix::identity().unwrap(), material: Material::default() }
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
    let ray2 = ray.transform(&self.transform.inverse().unwrap());
    
    let sphere_to_ray = &ray2.origin - &self.origin;
    
    let a = ray2.direction.dot(&ray2.direction);
    let b = 2.0 * ray2.direction.dot(&sphere_to_ray);
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
    let sphere = Sphere::new();

    let ray = Ray{
      origin: Vector::from([0.0, 0.0, -5.0, 1.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };
    let intersections = sphere.intersections(&ray);
    println!("{:?}", intersections);
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
      origin: Vector::from([0.0, 1.0, -5.0, 1.0]),
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

  #[test]
  fn test_scaled_sphere_intersection() {
    let ray = Ray{
      origin: Vector::from([0.0, 0.0, -5.0, 1.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };

    let mut sphere = Sphere::new();
    sphere.transform = sphere.transform.scale(2.0, 2.0, 2.0);
    assert_eq!(sphere.transform, Matrix::scaling(2.0, 2.0, 2.0));

    let intersections = sphere.intersections(&ray);
    assert_eq!(intersections.len(), 2);
    assert_eq!(intersections[0].t, 3.0);
    assert_eq!(intersections[1].t, 7.0);
  }

  #[test]
  fn test_translated_sphere_intersection() {
    let ray = Ray{
      origin: Vector::from([0.0, 0.0, -5.0, 1.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };

    let mut sphere = Sphere::new();
    sphere.transform = sphere.transform.translate(5.0, 0.0, 0.0);

    let intersections: Vec<Intersection<Sphere>> = sphere.intersections(&ray);
    assert_eq!(intersections.len(), 0);

    let ray = Ray{
      origin: Vector::from([0.0, 0.0, -5.0, 1.0]),
      direction: Vector::from([0.0, 0.0, 1.0, 0.0]),
    };

    let mut sphere = Sphere::new();
    sphere.transform = sphere.transform.translate(0.1, 0.0, 0.0);

    let intersections: Vec<Intersection<Sphere>> = sphere.intersections(&ray);
    assert_eq!(intersections.len(), 2);
  }

  #[test]
  fn test_normal_at() {
    let sphere = Sphere::new();
    let normal = sphere.normal_at(&Vector::from([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(normal, Vector::from([1.0, 0.0, 0.0, 0.0]));

    let normal = sphere.normal_at(&Vector::from([0.0, 1.0, 0.0, 1.0]));
    assert_eq!(normal, Vector::from([0.0, 1.0, 0.0, 0.0]));

    let normal = sphere.normal_at(&Vector::from([0.0, 0.0, 1.0, 1.0]));
    assert_eq!(normal, Vector::from([0.0, 0.0, 1.0, 0.0]));
  }

  #[test]
  fn test_normal_at_translated() {
    let mut sphere = Sphere::new();
    sphere.transform = sphere.transform.translate(0.0, 1.0, 0.0);
    let normal = sphere.normal_at(&Vector::from([1.0, 0.0, 0.0, 1.0]));
    assert_eq!(normal, Vector::from([0.7071067811865475, 0.7071067811865475, 0.0, 0.0]));

    // sphere.transform = Matrix::scaling(1.0, 0.5, 1.0).rotate_z(std::f64::consts::PI / 5.0);
    sphere.transform = &Matrix::scaling(1.0, 0.5, 1.0) * &Matrix::rotation_z(std::f64::consts::PI / 5.0);
    let normal = sphere.normal_at(&Vector::from([0.0, ((2.0 as f64).sqrt())/2.0, -((2.0 as f64).sqrt())/2.0, 0.0]));
    assert_eq!(normal, Vector::from([0.0, 0.9701425001453319, -0.24253562503633305, 0.0]));
  }
}