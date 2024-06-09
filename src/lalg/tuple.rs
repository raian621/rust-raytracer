use std::ops::{Add, Index, IndexMut, Neg, Sub};

use super::traits::{Addable, Dividable, Multipliable, Negateable, Subtractable};

#[derive(Debug, Clone, PartialEq)]
pub struct Tuple<T, const size: usize> {
  pub data: [T; size]
}

pub trait Coordinates<T: Clone> {
  fn x(&self) -> &T;
  fn y(&self) -> &T;
  fn z(&self) -> &T;
  fn w(&self) -> &T;
}

impl<T: Clone, const size: usize> Coordinates<T> for Tuple<T, size> {
  fn x(&self) -> &T { &self[0] }
  fn y(&self) -> &T { &self[1] }
  fn z(&self) -> &T { &self[2] }
  fn w(&self) -> &T { &self[3] }
}

impl<T: Default + Copy + Clone, const size: usize> Tuple<T, size> {
  pub fn new() -> Self {
    Self { data: [T::default(); size] }
  }

  pub fn from(data: [T; size]) -> Self { Self { data } }

  pub fn with_dimension() -> Self {
    Self { data: [T::default(); size] }
  }
}

impl<T: Multipliable<T>, const size: usize> Tuple<T, size> {
  pub fn scalar_mul(&self, scalar: &T) -> Self {
    let mut result = [T::default(); size];
    for i in 0..size {
      result[i] = result[i] * *scalar;
    }
    Tuple::from(result)
  }
}

impl<T: Dividable<T>, const size: usize> Tuple<T, size> {
  pub fn scalar_div(&self, scalar: &T) -> Self {
    let mut result = [T::default(); size];
    for i in 0..size {
      result[i] = result[i] / *scalar;
    }
    Tuple::from(result)
  }
}

impl<T: Addable<T>, const size: usize> Add for &Tuple<T, size> {
  type Output = Option<Tuple<T, size>>;

  fn add(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None
    }

    let mut result = Tuple::from([T::default(); size]);
    for i in 0..self.data.len() {
      result[i] = self[i].clone() + other[i].clone();
    }

    Some(result)
  }
}

impl<T: Subtractable<T>, const size: usize> Sub for &Tuple<T, size> {
  type Output = Option<Tuple<T, size>>;

  fn sub(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None
    }

    let mut result = Tuple::with_dimension();
    for i in 0..self.data.len() {
      result[i] = self[i].clone() - other[i].clone();
    }

    Some(result)
  }
}

impl<T: Negateable<T>, const size: usize> Neg for &Tuple<T, size> {
  type Output = Tuple<T, size>;

  fn neg(self) -> Self::Output {
    let mut result = Tuple::with_dimension();
    for i in 0..self.data.len() {
      result[i] = -self[i].clone();
    }
    
    result
  }
}

impl<T, const size: usize> Index<usize> for Tuple<T, size> {
  type Output = T;

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<T, const size: usize> IndexMut<usize> for Tuple<T, size> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T { &mut self.data[index] }
}

impl<T, const size: usize> Tuple<T, size> {
  pub fn len(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add() {
    let vec1 = Tuple::from([1, 2, 3]);
    let vec2 = Tuple::from([4, 8, 5]);
    let expected = Tuple::from([5, 10, 8]);
    let sum = (&vec1 + &vec2).unwrap();

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let vec1 = Tuple::from([1, 2, 3]);
    let vec2 = Tuple::from([4, 8, 5]);
    let expected = Tuple::from([-3, -6, -2]);
    let diff = (&vec1 - &vec2).unwrap();

    assert_eq!(diff, expected);
  }

  #[test]
  fn test_neg() {
    let vec1 = Tuple::from([1, 2, 3]);
    let expected = Tuple::from([-1, -2, -3]);
    let neg = -&vec1;

    assert_eq!(neg, expected);
  }

  #[test]
  fn test_scalar_mul() {
    let vec1 = Tuple::from([1, 2, 3]);
    let expected = Tuple::from([2, 4, 6]);
    let scalar_mul = vec1.scalar_mul(&2);

    assert_eq!(scalar_mul, expected);
  }

  #[test]
  fn test_coords() {
    let (x, y, z, w) = (0, 1, 2, 3);
    let vec1 = Tuple::from([x, y, z, w]);

    assert_eq!(x, *vec1.x());
    assert_eq!(y, *vec1.y());
    assert_eq!(z, *vec1.z());
    assert_eq!(w, *vec1.w());
  }
}