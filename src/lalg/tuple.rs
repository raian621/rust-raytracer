use std::ops::{Add, Index, IndexMut, Neg, Sub};

use super::traits::{Addable, Dividable, Multipliable, Negateable, Subtractable};

#[derive(Debug, Clone, PartialEq)]
pub struct Tuple<T> {
  pub data: Vec<T>
}

pub trait Coordinates<T: Clone> {
  fn x(&self) -> &T;
  fn y(&self) -> &T;
  fn z(&self) -> &T;
  fn w(&self) -> &T;
}

impl<T: Clone> Coordinates<T> for Tuple<T> {
  fn x(&self) -> &T { &self[0] }
  fn y(&self) -> &T { &self[1] }
  fn z(&self) -> &T { &self[2] }
  fn w(&self) -> &T { &self[3] }
}

impl<T: Default + Clone> Tuple<T> {
  pub fn new() -> Tuple<T> {
    Self { data: Vec::<T>::new() }
  }

  pub fn from(data: Vec<T>) -> Self { Self { data } }

  pub fn with_dimension(dimension: usize) -> Self {
    Self { data: vec![T::default(); dimension] }
  }
}

impl<T: Multipliable<T>> Tuple<T> {
  pub fn scalar_mul(&self, scalar: &T) -> Self {    
    let result = self.data.iter().map(|val| *val * *scalar).collect::<Vec<T>>();
    Tuple::from(result)
  }
}

impl<T: Dividable<T>> Tuple<T> {
  pub fn scalar_div(&self, scalar: &T) -> Self {
    let result = self.data.iter().map(|val| *val / *scalar).collect::<Vec<T>>();
    Tuple::from(result)
  }
}

impl<T: Addable<T>> Add for &Tuple<T> {
  type Output = Option<Tuple<T>>;

  fn add(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None
    }

    let mut result: Tuple<T> = Tuple::with_dimension(self.data.len());
    for i in 0..self.data.len() {
      result[i] = self[i].clone() + other[i].clone();
    }

    Some(result)
  }
}

impl<T: Subtractable<T>> Sub for &Tuple<T> {
  type Output = Option<Tuple<T>>;

  fn sub(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None
    }

    let mut result: Tuple<T> = Tuple::with_dimension(self.data.len());
    for i in 0..self.data.len() {
      result[i] = self[i].clone() - other[i].clone();
    }

    Some(result)
  }
}

impl<T: Negateable<T>> Neg for &Tuple<T> {
  type Output = Tuple<T>;

  fn neg(self) -> Self::Output {
    let mut result: Tuple<T> = Tuple::with_dimension(self.data.len());
    for i in 0..self.data.len() {
      result[i] = -self[i].clone();
    }
    
    result
  }
}

impl<T> Index<usize> for Tuple<T> {
  type Output = T;

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<T> IndexMut<usize> for Tuple<T> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut T { &mut self.data[index] }
}

impl<T> Tuple<T> {
  pub fn len(&self) -> usize { self.data.len() }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_add() {
    let vec1 = Tuple::from(vec![1, 2, 3]);
    let vec2 = Tuple::from(vec![4, 8, 5]);
    let expected = Tuple::from(vec![5, 10, 8]);
    let sum = (&vec1 + &vec2).unwrap();

    assert_eq!(sum, expected);
    
    let vec2 = Tuple::from(vec![4, 8]);
    let expected: Option<Tuple<i32>> = None;
    let sum = &vec1 + &vec2;
    
    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let vec1 = Tuple::from(vec![1, 2, 3]);
    let vec2 = Tuple::from(vec![4, 8, 5]);
    let expected = Tuple::from(vec![-3, -6, -2]);
    let diff = (&vec1 - &vec2).unwrap();

    assert_eq!(diff, expected);
    
    let vec2 = Tuple::from(vec![4, 8]);
    let expected: Option<Tuple<i32>> = None;
    let diff = &vec1 - &vec2;
    
    assert_eq!(diff, expected);
  }

  #[test]
  fn test_neg() {
    let vec1 = Tuple::from(vec![1, 2, 3]);
    let expected = Tuple::from(vec![-1, -2, -3]);
    let neg = -&vec1;

    assert_eq!(neg, expected);
  }

  #[test]
  fn test_scalar_mul() {
    let vec1 = Tuple::from(vec![1, 2, 3]);
    let expected = Tuple::from(vec![2, 4, 6]);
    let scalar_mul = vec1.scalar_mul(&2);

    assert_eq!(scalar_mul, expected);
  }

  #[test]
  fn test_coords() {
    let (x, y, z, w) = (0, 1, 2, 3);
    let vec1 = Tuple::from(vec![x, y, z, w]);

    assert_eq!(x, *vec1.x());
    assert_eq!(y, *vec1.y());
    assert_eq!(z, *vec1.z());
    assert_eq!(w, *vec1.w());
  }
}