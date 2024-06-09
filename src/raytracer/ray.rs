use crate::lalg::vector::Vector;

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
}

pub trait Intersects {
  fn intersections(&self, ray: Ray) -> Vec<f64>;
}

#[cfg(tests)]
pub mod tests {
  use super::*;
}