use crate::lalg::vector::Vector;

use super::intersection::Intersection;

pub struct Ray {
  pub origin: Vector<f64>,
  pub direction: Vector<f64>
}

impl Ray {
  pub fn new() -> Ray {
    Ray {
      origin: Vector::with_dimension(4),
      direction: Vector::with_dimension(4)
    }
  }

  pub fn with(origin: Vector<f64>, direction: Vector<f64>) -> Ray {
    Ray { origin, direction }
  }

  pub fn get_point(&self, t: f64) -> Vector<f64> {
    (&self.origin + &self.direction.scalar_mul(&t)).unwrap()
  }
}

pub trait Intersects<T> {
  fn intersections<'a>(&'a self, ray: &Ray) -> Vec<Intersection<'a, T>>;
}

#[cfg(test)]
pub mod tests {
  use super::*;

  #[test]
  fn test_get_point() {
    let ray = Ray{
      origin:    Vector::from(vec![0.0,  0.0, 0.0, 0.0]),
      direction: Vector::from(vec![1.0, -1.0, 0.0, 1.0])
    };
    let t = 5.0;
    let expected = Vector::from(vec![5.0, -5.0, 0.0, 5.0]);
    let point = ray.get_point(t);

    assert_eq!(point, expected);

    let ray = Ray{
      origin:    Vector::from(vec![2.0, 5.0, 3.0, 0.0]),
      direction: Vector::from(vec![1.0, -1.0, 0.0, 1.0])
    };
    let t = 5.0;
    let expected = Vector::from(vec![7.0, 0.0, 3.0, 5.0]);
    let point = ray.get_point(t);

    assert_eq!(point, expected);
  }
}