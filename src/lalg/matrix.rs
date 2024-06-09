use std::ops::{Add, Index, IndexMut, Mul, Sub};

use super::{traits::{Addable, Subtractable}, tuple::{Tuple}};

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

impl Matrix<f64> {
  pub fn identity(dimension: usize) -> Result<Self, String> {
    let mut result = Self::with_dimensions(dimension, dimension)?;

    for i in 0..dimension {
      result[i][i] = 1.0;
    }

    Ok(result)
  }

  pub fn transpose(&self) -> Self {
    let mut result = Self::with_dimensions(self.num_cols(), self.num_rows()).unwrap();

    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[col][row] = self[row][col].clone();
      }
    }

    result
  }
}

impl Matrix<f64> {
  pub fn scalar_mul(&self, scalar: f64) -> Matrix<f64> {
    let mut result = self.clone();
    
    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[row][col] *= scalar;
      }
    }

    result
  }

  pub fn scalar_div(&self, scalar: f64) -> Matrix<f64> {
    let mut result = self.clone();
    
    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[row][col] /= scalar;
      }
    }

    result
  }

  /*
   this determinant algorithm gets the upper triangular form
   of the input matrix and multiplies the diagonal values of
   the upper triangular matrix to get the determinant

   time ->  O(n^3)
   space -> O(n^2)

   if we transformed the matrix in-place the space complexity
   would be O(1) though
   */
  pub fn determinant(&self) -> Result<f64, String> {
    if self.num_cols() != self.num_rows() {
      return Err("cannot get the determinant of a non-square matrix".to_string());
    }

    let mut upper_triangular = self.clone();
    for row in 1..self.num_rows() {
      for col in 0..row {
        // avoid divide-by-zero errors
        if upper_triangular[row][col] == 0.0 {
          continue;
        }
        let factor = upper_triangular[row][col] / upper_triangular[col][col];
        for k in 0..self.num_cols() {
          upper_triangular[row][k] -= factor * upper_triangular[col][k];
        }
      }
    }

    let mut det = 1 as f64;
    for i in 0..self.num_rows() {
      det *= upper_triangular[i][i];
    }

    Ok(det)
  }

  pub fn submatrix(&self, row: usize, col: usize) -> Result<Self, String> {
    if row >= self.num_rows() || col >= self.num_cols() {
      return Err(format!("submatrix does not exist for row={}, col={}", row, col));
    }

    let mut submatrix = Self::with_dimensions(self.num_rows()-1, self.num_cols()-1).unwrap();
    let mut subrow = 0;

    for r in 0..self.num_rows() {
      if r == row {
        continue;
      }
      let mut subcol = 0;

      for c in 0..self.num_cols() {
        if c == col {
          continue;
        }

        submatrix[subrow][subcol] = self[r][c];
        subcol += 1;
      }

      subrow += 1;
    }

    Ok(submatrix)
  }

  pub fn minor(&self, row: usize, col: usize) -> Result<f64, String> {
    let submatrix = self.submatrix(row, col)?;
    Ok(submatrix.determinant()?)
  }

  pub fn cofactor(&self, row: usize, col: usize) -> Result<f64, String> {
    match (row + col) % 2 {
      0 => Ok(self.minor(row, col)?),
      _ => Ok(-self.minor(row, col)?)
    }
  }

  pub fn inverse(&self) -> Result<Self, String> {
    let determinant = self.determinant()?;
    if determinant == 0.0 {
      return Err("cannot invert a matrix if it's determinant is zero".to_string());
    }

    let mut result = self.transpose();
    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[col][row] = self.cofactor(row, col)? / determinant;
      }
    }

    Ok(result)
  }

  pub fn translating(x: f64, y: f64, z: f64) -> Self {
    Self::from(vec![
      vec![1.0, 0.0, 0.0, x],
      vec![0.0, 1.0, 0.0, y],
      vec![0.0, 0.0, 1.0, z],
      vec![0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn scaling(sx: f64, sy: f64, sz: f64) -> Self {
    Self::from(vec![
      vec![ sx, 0.0, 0.0, 0.0],
      vec![0.0,  sy, 0.0, 0.0],
      vec![0.0, 0.0,  sz, 0.0],
      vec![0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn rotation_x(r: f64) -> Self {
    Self::from(vec![
      vec![1.0,     0.0,      0.0, 0.0],
      vec![0.0, r.cos(), -r.sin(), 0.0],
      vec![0.0, r.sin(),  r.cos(), 0.0],
      vec![0.0,     0.0,      0.0, 1.0],
    ])
  }

  pub fn rotation_y(r: f64) -> Self {
    Self::from(vec![
      vec![r.cos(),  0.0, r.sin(), 0.0],
      vec![0.0,      1.0,     0.0, 0.0],
      vec![-r.sin(), 0.0, r.cos(), 0.0],
      vec![0.0,      0.0,     0.0, 1.0],
    ])
  }

  pub fn rotation_z(r: f64) -> Self {
    Self::from(vec![
      vec![r.cos(), -r.sin(), 0.0, 0.0],
      vec![r.sin(),  r.cos(), 0.0, 0.0],
      vec![0.0,          0.0, 1.0, 0.0],
      vec![0.0,          0.0, 0.0, 1.0],
    ])
  }

  pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
    Self::from(vec![
      vec![1.0,  xy,  xz, 0.0],
      vec![ yx, 1.0,  yz, 0.0],
      vec![ zx,  zy, 1.0, 0.0],
      vec![0.0, 0.0, 0.0, 1.0]
    ])
  }

  pub fn translate(&self, x: f64, y: f64, z: f64) -> Self {
    (&Self::translating(x, y, z) * self).unwrap()
  }

  pub fn scale(&self, x: f64, y: f64, z: f64) -> Self {
    (&Self::scaling(x, y, z) * self).unwrap()
  }

  pub fn rotate_x(&self, r: f64) -> Self {
    (&Self::rotation_x(r) * self).unwrap()
  }

  pub fn rotate_y(&self, r: f64) -> Self {
    (&Self::rotation_y(r) * self).unwrap()
  }

  pub fn rotate_z(&self, r: f64) -> Self {
    (&Self::rotation_z(r) * self).unwrap()
  }

  pub fn shear(&self, xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
    (&Self::shearing(xy, xz, yx, yz, zx, zy) * self).unwrap()
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

  #[test]
  fn test_transpose() {
    let mat = Matrix::<f64>::from(vec![
      vec![1.0, 2.0, 3.0],
      vec![4.0, 5.0, 6.0],
      vec![7.0, 8.0, 9.0],
    ]);
    let expected = Matrix::<f64>::from(vec![
      vec![1.0, 4.0, 7.0],
      vec![2.0, 5.0, 8.0],
      vec![3.0, 6.0, 9.0],
    ]);
    let transpose = mat.transpose();

    assert_eq!(transpose, expected);

    let mat = Matrix::<f64>::from(vec![
      vec![1.0, 2.0, 3.0, 1.0],
      vec![4.0, 5.0, 6.0, 2.0],
      vec![7.0, 8.0, 9.0, 3.0],
    ]);
    let expected = Matrix::<f64>::from(vec![
      vec![1.0, 4.0, 7.0],
      vec![2.0, 5.0, 8.0],
      vec![3.0, 6.0, 9.0],
      vec![1.0, 2.0, 3.0],
    ]);
    let transpose = mat.transpose();

    assert_eq!(transpose, expected);
  }

  #[test]
  fn test_scalar_mul() {
    let mat = Matrix::<f64>::from(vec![
      vec![1.0, 2.0],
      vec![3.0, 4.0],
    ]);
    let product = mat.scalar_mul(2.0);
    let expected = Matrix::<f64>::from(vec![
      vec![2.0, 4.0],
      vec![6.0, 8.0],
    ]);

    assert_eq!(product, expected);
  }

  #[test]
  fn test_determinant() {
    let mat = Matrix::<f64>::from(vec![
      vec![-2.0, -8.0,  3.0,  5.0],
      vec![-3.0,  1.0,  7.0,  3.0],
      vec![ 1.0,  2.0, -9.0,  6.0],
      vec![-6.0,  7.0,  7.0, -9.0],
    ]);
    let determinant = mat.determinant().unwrap();
    let expected = -4071 as f64;

    assert_eq!(determinant, expected);

    let mat = Matrix::<f64>::from(vec![
      vec![ 1.0,  2.0,  6.0],
      vec![-5.0,  8.0, -4.0],
      vec![ 2.0,  6.0,  4.0],
    ]);
    let determinant = mat.determinant().unwrap();
    let expected = -196 as f64;

    assert_eq!(determinant, expected);
  }

  #[test]
  fn test_inverse() {
    let mat = Matrix::<f64>::from(vec![
      vec![-5.0,  2.0,  6.0, -8.0],
      vec![ 1.0, -5.0,  1.0,  8.0],
      vec![ 7.0,  7.0, -6.0, -7.0],
      vec![ 1.0, -3.0,  7.0,  4.0],
    ]);
    let expected = Matrix::<f64>::from(vec![
      vec![ 0.21805,  0.45113,  0.24060, -0.04511],
      vec![-0.80827, -1.45677, -0.44361,  0.52068],
      vec![-0.07895, -0.22368, -0.05263,  0.19737],
      vec![-0.52256, -0.81391, -0.30075,  0.30639],
    ]);
    let inverse = mat.inverse().unwrap();
    let diff = (&inverse - &expected).unwrap();
    for row in 0..inverse.num_rows() {
      for col in 0..inverse.num_cols() {
        assert!(diff[row][col].abs() < 1e-5)
      }
    }
  }
}
