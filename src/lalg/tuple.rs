use std::{fmt::Debug, ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub}};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tuple<const SIZE: usize> {
  pub data: [f64; SIZE]
}

pub trait Coordinates<T: Clone> {
  fn x(&self) -> &T;
  fn y(&self) -> &T;
  fn z(&self) -> &T;
  fn w(&self) -> &T;
}

impl<const SIZE: usize> Coordinates<f64> for Tuple<SIZE> {
  fn x(&self) -> &f64 { &self[0] }
  fn y(&self) -> &f64 { &self[1] }
  fn z(&self) -> &f64 { &self[2] }
  fn w(&self) -> &f64 { &self[3] }
}

impl<const SIZE: usize> Tuple<SIZE> {
  pub fn new() -> Self {
    Self { data: [0.0; SIZE] }
  }

  pub fn from(data: [f64; SIZE]) -> Self { Self { data: data.clone() } }

  pub fn with_dimension() -> Self {
    Self { data: [0.0; SIZE] }
  }
  
  pub fn scalar_mul(&self, scalar: &f64) -> Self {
    let mut result = [0.0; SIZE];
    for i in 0..SIZE {
      result[i] = self[i] * *scalar;
    }
    Tuple::from(result)
  }

  pub fn scalar_div(&self, scalar: &f64) -> Self {
    let mut result = [0.0; SIZE];
    for i in 0..SIZE {
      result[i] = self[i] / *scalar;
    }
    Tuple::from(result)
  }

  pub fn len(&self) -> usize { self.data.len() }
}

impl<const SIZE: usize> Index<usize> for Tuple<SIZE> {
  type Output = f64;

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<const SIZE: usize> IndexMut<usize> for Tuple<SIZE> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut f64 { &mut self.data[index] }
}

impl<const SIZE: usize> Add<Self> for &Tuple<SIZE> {
  type Output = Tuple<SIZE>;

  fn add(self, other: Self) -> Self::Output {
    let mut result = Tuple::from([0.0; SIZE]);
    for i in 0..self.data.len() {
      result[i] = self[i].clone() + other[i].clone();
    }

    result
  }
}

impl<const SIZE: usize> Sub<Self> for &Tuple<SIZE> {
  type Output = Tuple<SIZE>;

  fn sub(self, other: Self) -> Self::Output {
    let mut result = Tuple::from([0.0; SIZE]);
    for i in 0..self.data.len() {
      result[i] = self[i].clone() - other[i].clone();
    }

    result
  }
}

impl<const SIZE: usize> Mul<Self> for &Tuple<SIZE> {
  type Output = Tuple<SIZE>;

  fn mul(self, other: Self) -> Self::Output {
    let mut result = Tuple::from([0.0; SIZE]);
    for i in 0..self.data.len() {
      result[i] = self[i].clone() * other[i].clone();
    }

    result
  }
}

impl<const SIZE: usize> Div<Self> for &Tuple<SIZE> {
  type Output = Tuple<SIZE>;

  fn div(self, other: Self) -> Self::Output {
    let mut result = Tuple::from([0.0; SIZE]);
    for i in 0..self.data.len() {
      result[i] = self[i].clone() + other[i].clone();
    }

    result
  }
}

impl<const SIZE: usize> Neg for &Tuple<SIZE> {
  type Output = Tuple<SIZE>;

  fn neg(self) -> Self::Output {
    let mut result = Tuple::from([0.0; SIZE]);
    for i in 0..self.data.len() {
      result[i] = -self[i].clone()
    }

    result
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add() {
    let vec1 = Tuple::from([1.0, 2.0, 3.0]);
    let vec2 = Tuple::from([4.0, 8.0, 5.0]);
    let expected = Tuple::from([5.0, 10.0, 8.0]);
    let sum = &vec1 + &vec2;

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let vec1 = Tuple::from([1.0, 2.0, 3.0]);
    let vec2 = Tuple::from([4.0, 8.0, 5.0]);
    let expected = Tuple::from([-3.0, -6.0, -2.0]);
    let diff = &vec1 - &vec2;

    assert_eq!(diff, expected);
  }

  #[test]
  fn test_neg() {
    let vec1 = Tuple::from([1.0, 2.0, 3.0]);
    let expected = Tuple::from([-1.0, -2.0, -3.0]);
    let neg = -&vec1;

    assert_eq!(neg, expected);
  }

  #[test]
  fn test_scalar_mul() {
    let vec1 = Tuple::from([1.0, 2.0, 3.0]);
    let expected = Tuple::from([2.0, 4.0, 6.0]);
    let scalar_mul = vec1.scalar_mul(&2.0);

    assert_eq!(scalar_mul, expected);
  }

  #[test]
  fn test_coords() {
    let (x, y, z, w) = (0.0, 1.0, 2.0, 3.0);
    let vec1 = Tuple::from([x, y, z, w]);

    assert_eq!(x, *vec1.x());
    assert_eq!(y, *vec1.y());
    assert_eq!(z, *vec1.z());
    assert_eq!(w, *vec1.w());
  }
}