use std::ops::{Add, Index, IndexMut, Mul, Sub};

use super::{traits::{Addable, Subtractable}, tuple::Tuple};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
  pub data: Vec<Vec<T>>
}

impl<T> Matrix<T> {
  pub fn num_rows(&self) -> usize { self.data.len() }
  pub fn num_cols(&self) -> usize { self.data[0].len() }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
  fn from(data: Vec<Vec<T>>) -> Matrix<T> { Matrix{ data }}
}

impl<T: Clone + Default> From<Tuple<T>> for Matrix<T> {
  fn from(data: Tuple<T>) -> Matrix<T> {
    let mut result = Matrix::<T>::with_dimensions(data.len(), 1).unwrap();
    for (i, val) in data.data.iter().enumerate() {
      result[i][0] = val.clone();
    }

    result
  }
}

impl<T: Default + Clone> Matrix<T> {
  pub fn new() -> Matrix<T> { Matrix{ data: Vec::<Vec<T>>::new()}}
  
  pub fn with_dimensions(rows: usize, cols: usize) -> Result<Matrix<T>, String> {
    if rows == 0 {
      return Err("can't create a matrix with 0 rows".to_string());
    }

    Ok(Matrix::from(
      vec![vec![T::default(); cols]; rows]
    ))
  }
}

impl<T> Index<usize> for Matrix<T> {
  type Output = Vec<T>;

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<T> IndexMut<usize> for Matrix<T> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut Vec<T> { &mut self.data[index] }
}

impl<T: Addable<T> + Copy> Add for &Matrix<T> {
  type Output = Option<Matrix<T>>;

  fn add(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None;
    }

    let mut matrix: Matrix<T> = Matrix::with_dimensions(self.num_rows(), self.num_cols()).unwrap();

    for row in 0..matrix.num_rows() {
      for col in 0..matrix.num_cols() {
        matrix[row][col] = self[row][col] + other[row][col];
      }
    }

    Some(matrix)
  }
}

impl<T: Subtractable<T> + Copy> Sub for &Matrix<T> {
  type Output = Option<Matrix<T>>;

  fn sub(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None;
    }

    let mut matrix: Matrix<T> = Matrix::with_dimensions(self.num_rows(), self.num_cols()).unwrap();

    for row in 0..matrix.num_rows() {
      for col in 0..matrix.num_cols() {
        matrix[row][col] = self[row][col] - other[row][col];
      }
    }

    Some(matrix)
  }
}

impl Mul<Self> for &Matrix<f64> {
  type Output = Result<Matrix<f64>, String>;

  fn mul(self, other: Self) -> Self::Output {
    let (rows, cols, shared) = (self.num_rows(), other.num_cols(), self.num_cols());
    if shared != other.num_rows() {
      return Err("the number of columns in the len matrix must be the same as the number of rows in the right matrix".to_string());
    }

    let mut result = Matrix::<f64>::with_dimensions(self.num_rows(), other.num_cols())?;

    for m in 0..rows {
      for n in 0..cols {
        for k in 0..shared {
          result[m][n] += self[m][k] * other[k][n];
        }
      }
    }

    Ok(result)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_with_dimensions() {
    let rows = 2;
    let cols = 3;
    let result: Result<Matrix<i32>, String> = Matrix::with_dimensions(rows, cols);

    let matrix = match result {
      Ok(val) => val,
      Err(why) => { panic!("{}", why);}
    };
    
    for row in 0..rows {
      for col in 0..cols {
        assert_eq!(matrix[row][col], 0);
      }
    }
  }

  #[test]
  fn test_add() {
    let mat1 = Matrix::from(vec![
      vec![1, 2, 3],
      vec![4, 5, 6],
      vec![7, 8, 9],
    ]);
    let mat2 = Matrix::from(vec![
      vec![9, 8, 7],
      vec![6, 5, 4],
      vec![3, 2, 1],
    ]);
    let result = &mat1 + &mat2;
    let sum = match result {
      Some(val) => val,
      None => panic!()
    };
    let expected = Matrix::from(vec![
      vec![10, 10, 10],
      vec![10, 10, 10],
      vec![10, 10, 10],
    ]);

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let mat1 = Matrix::from(vec![
      vec![1, 2, 3],
      vec![4, 5, 6],
      vec![7, 8, 9],
    ]);
    let mat2 = Matrix::from(vec![
      vec![9, 8, 7],
      vec![6, 5, 4],
      vec![3, 2, 1],
    ]);
    let result = &mat1 - &mat2;
    let diff = match result {
      Some(val) => val,
      None => panic!()
    };
    let expected = Matrix::from(vec![
      vec![-8, -6, -4],
      vec![-2, 0, 2],
      vec![4, 6, 8],
    ]);

    assert_eq!(diff, expected);
  }

  #[test]
  fn test_mul() {
    let mat1 = Matrix::<f64>::from(vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![5.0, 6.0, 7.0, 8.0],
      vec![9.0, 8.0, 7.0, 6.0],
      vec![5.0, 4.0, 3.0, 2.0],
    ]);
    let mat2 = Matrix::<f64>::from(vec![
      vec![-2.0, 1.0, 2.0,  3.0],
      vec![ 3.0, 2.0, 1.0, -1.0],
      vec![ 4.0, 3.0, 6.0,  5.0],
      vec![ 1.0, 2.0, 7.0,  8.0],
    ]);
    let expected = Matrix::<f64>::from(vec![
      vec![20.0, 22.0,  50.0,  48.0],
      vec![44.0, 54.0, 114.0, 108.0],
      vec![40.0, 58.0, 110.0, 102.0],
      vec![16.0, 26.0,  46.0,  42.0],
    ]);
    let mul = match &mat1 * &mat2 {
      Err(why) => panic!("{}", why),
      Ok(val) => val
    };

    assert_eq!(expected, mul);
  }

  #[test]
  fn test_tuple_mul() {
    let mat = Matrix::from(vec![
      vec![1.0, 2.0, 3.0, 4.0],
      vec![2.0, 4.0, 4.0, 2.0],
      vec![8.0, 6.0, 4.0, 1.0],
      vec![0.0, 0.0, 0.0, 1.0],
    ]);
    let tuple = Tuple::from(vec![1.0, 2.0, 3.0, 1.0]);
    let expected = Matrix::from(Tuple::from(vec![18.0, 24.0, 33.0, 1.0]));
    let product = (&mat * &Matrix::from(tuple)).unwrap();
  
    assert_eq!(expected, product);
  }
}
