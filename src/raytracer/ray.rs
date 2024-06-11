use crate::lalg::{matrix::Matrix, vector::Vector};

#[derive(Debug, Clone)]
pub struct Ray {
  pub origin: Vector<4>,
  pub direction: Vector<4>
}

impl Ray {
  pub fn new() -> Ray {
    Ray {
      origin: Vector::from([0.0, 0.0, 0.0, 1.0]),
      direction: Vector::with_dimension()
    }
  }

  pub fn with(origin: Vector<4>, direction: Vector<4>) -> Ray {
    Ray { origin, direction }
  }

  pub fn get_point(&self, t: f64) -> Vector<4> {
    &self.origin + &self.direction.scalar_mul(&t)
  }

  pub fn transform(&self, transform: &Matrix<4, 4>) -> Self {
    let origin = transform * &Matrix::from(self.origin.clone());
    let direction = transform * &Matrix::from(self.direction.clone());

    Ray{ origin: Vector::from(&origin), direction: Vector::from(&direction)}
  }
}

#[cfg(test)]
pub mod tests {
  use super::*;

  #[test]
  fn test_get_point() {
    let ray = Ray{
      origin:    Vector::from([0.0,  0.0, 0.0, 1.0]),
      direction: Vector::from([1.0, -1.0, 0.0, 0.0])
    };
    let t = 5.0;
    let expected = Vector::from([5.0, -5.0, 0.0, 1.0]);
    let point = ray.get_point(t);

    assert_eq!(point, expected);

    let ray = Ray{
      origin:    Vector::from([2.0, 5.0, 3.0, 0.0]),
      direction: Vector::from([1.0, -1.0, 0.0, 1.0])
    };
    let t = 5.0;
    let expected = Vector::from([7.0, 0.0, 3.0, 5.0]);
    let point = ray.get_point(t);

    assert_eq!(point, expected);
  }
}