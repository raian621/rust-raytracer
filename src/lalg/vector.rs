use std::ops::{Div, Mul};

use super::tuple::Tuple;

pub type Vector<T> = Tuple<T>;

impl<T: Clone + Into<f64>> Vector<T> {
  pub fn dot(&self, other: &Self) -> Option<f64> {
    if self.len() != other.len() {
      return None;
    }

    let mut product = 0.0;
    for i in 0..self.len() {
      product = product + self[i].clone().into() * other[i].clone().into();
    }

    Some(product)
  }

  pub fn magnitude(&self) -> f64 {
    let dot: f64 =  self.dot(self).unwrap().into();
    dot.sqrt()
  }
}

impl<T: Default + Clone + Copy + Div<Output = T> + Mul<Output = T> + Into<f64> + From<f64>> Vector<T> {
  pub fn norm(&self) -> Self {
    let magnitude = self.magnitude();
    self.scalar_div(&T::from(magnitude))
  }

  pub fn cross(&self, other: &Self) -> Option<Vector<f64>>{
    if self.len() < 3 || other.len() < 3 {
      return None;
    }

    let mut cross: Vector<f64> = Vector::with_dimension(3);
    cross[0] = self[1].into() * other[2].into()
             - self[2].into() * other[1].into();
    cross[1] = self[2].into() * other[0].into()
             - self[0].into() * other[2].into();
    cross[2] = self[0].into() * other[1].into()
             - self[1].into() * other[0].into();

    Some(cross)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_dot() {
    let vec1 = Vector::from(vec![1, 2, 3]);
    let vec2 = Vector::from(vec![2, 3, 4]);
    let expected: Option<f64> = Some(20.0);
    let dot = vec1.dot(&vec2);

    assert_eq!(dot, expected);

    let vec1 = Vector::from(vec![1, 2, 3]);
    let vec2 = Vector::from(vec![2, 3,]);
    let expected: Option<f64> = None;
    let dot = vec1.dot(&vec2);

    assert_eq!(dot, expected);
  }

  #[test]
  fn test_magnitude() {
    let vec1 = Vector::from(vec![1, 2, 3]);
    let expected = (14.0 as f64).sqrt();
    let magnitude = vec1.magnitude();

    assert_eq!(expected, magnitude);
  }

  #[test]
  fn test_normalize() {
    let vec1: Vector<f64> = Vector::from(vec![1.0, 2.0, 3.0]);
    let magnitude = (14 as f64).sqrt();
    let expected = vec1.scalar_div(&magnitude);
    let normalized = vec1.norm();

    assert_eq!(expected, normalized);
  }

  #[test]
  fn test_cross() {
    let (x1, x2, y1, y2, z1, z2) = (1 as f64, 2 as f64, 3 as f64, 4 as f64, 5 as f64, 6 as f64);
    let vec1 = Vector::from(vec![x1, y1, z1]);
    let vec2 = Vector::from(vec![x2, y2, z2]);
    let cross = vec1.cross(&vec2);
    let expected = Some(Vector::from(vec![
      y1 * z2 - y2 * z1,
      x2 * z1 - x1 * z2,
      x1 * y2 - x2 * y1,
    ]));

    assert_eq!(expected, cross);

    let (x1, x2, y1, y2, z1, z2) = (2 as f64, 5 as f64, 7 as f64, 3 as f64, 5 as f64, 8 as f64);
    let vec1 = Vector::from(vec![x1, y1, z1]);
    let vec2 = Vector::from(vec![x2, y2, z2]);
    let cross = vec1.cross(&vec2);
    let expected: Option<Tuple<f64>> = Some(Vector::from(vec![
      y1 * z2 - y2 * z1,
      x2 * z1 - x1 * z2,
      x1 * y2 - x2 * y1,
    ]));
    assert_eq!(expected, cross);
    
    let vec2 = Vector::from(vec![0.0 as f64, 0.0 as f64]);
    let cross = vec1.cross(&vec2);

    assert_eq!(None, cross);
  }
}