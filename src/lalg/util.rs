use std::{fmt::Debug, ops::{Add, Div, Mul, Neg, Sub}};

pub trait Determinant<T>: Default + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32> + PartialEq<T>{}
impl<T> Determinant<T> for T where T: Default + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + From<i32> + PartialEq<T>{}

pub fn submatrix<T: Default + Clone>(
  matrix: &Vec<Vec<T>>,
  row: usize,
  col: usize
) -> Result<Vec<Vec<T>>, String> {
  let rows = matrix.len();
  let cols = matrix[0].len();

  if row >= rows || col >= cols {
    return Err(format!("cannot make a submatrix for row {} and col {} from matrix with {} rows and {} cols", row, col, rows, cols));
  }

  let mut submat = vec![vec![T::default(); matrix[0].len()-1]; matrix.len()-1];
  let mut subrow = 0;

  for r in 0..rows {
    if r == row {
      continue;
    }
    let mut subcol = 0;

    for c in 0..cols {
      if c == col {
        continue;
      }
      submat[subrow][subcol] = matrix[r][c].clone();
      subcol += 1;
    }
    subrow += 1;
  }

  Ok(submat)
}


pub fn cofactor<T: Determinant<T> + Neg<Output = T> + Debug>(matrix: &Vec<Vec<T>>, row: usize, col: usize) -> Result<T, String> {
  match (row + col) % 2 {
    0 => Ok(minor(matrix, row, col)?),
    _ => Ok(-minor(matrix, row, col)?)
    }
}
pub fn minor<T: Determinant<T> + Debug>(matrix: &Vec<Vec<T>>, row: usize, col: usize) -> Result<T, String> {
  let submat = submatrix(matrix, row, col)?;
  Ok(determinant(&submat)?)
}

pub fn determinant<T: Determinant<T>>(matrix: &Vec<Vec<T>>) -> Result<T, String> {
  let rows = matrix.len();
  let cols = matrix[0].len();
  if rows != cols {
    return Err("cannot get the determinant of a non-square matrix".to_string());
  }

  let mut upper_triangular = matrix.clone();
  for row in 1..rows {
    for col in 0..row {
      // avoid divide-by-zero errors
      if upper_triangular[row][col] == T::default() || upper_triangular[col][col] == T::default() {
        continue;
      }
      let factor = upper_triangular[row][col] / upper_triangular[col][col];
      for k in 0..cols {
        upper_triangular[row][k] = upper_triangular[row][k] - factor * upper_triangular[col][k];
      }
    }
  }

  let mut det = T::default() + T::from(1);
  for i in 0..rows {
    det = det * upper_triangular[i][i];
  }

  Ok(det)
}

#[cfg(test)]
pub mod tests {
  use super::*;

  #[test]
  fn test_determinant() {
    let mat = vec![
      vec![-2.0, -8.0,  3.0,  5.0],
      vec![-3.0,  1.0,  7.0,  3.0],
      vec![ 1.0,  2.0, -9.0,  6.0],
      vec![-6.0,  7.0,  7.0, -9.0],
    ];
    let det = determinant(&mat).unwrap();
    let expected = -4071 as f64;
    
    assert!((cofactor(&mat, 0, 0).unwrap() - 690.0).abs() < 1e-5);
    assert!((cofactor(&mat, 0, 1).unwrap() - 447.0).abs() < 1e-5);
    assert!((cofactor(&mat, 0, 2).unwrap() - 210.0).abs() < 1e-5);
    assert!((cofactor(&mat, 0, 3).unwrap() - 51.0).abs() < 1e-5);
    assert_eq!(det, expected);

    let mat = vec![
      vec![ 1.0,  2.0,  6.0],
      vec![-5.0,  8.0, -4.0],
      vec![ 2.0,  6.0,  4.0],
    ];
    let det = determinant(&mat).unwrap();
    let expected = -196 as f64;

    assert_eq!(det, expected);
  }
}
