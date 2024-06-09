use std::ops::{Add, Index, IndexMut, Mul, Sub};

use super::{traits::{Addable, Subtractable}, tuple::Tuple};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T, const rows: usize, const cols: usize> {
  pub data: [[T; cols]; rows]
}

impl<T, const rows: usize, const cols: usize> Matrix<T, rows, cols> {
  pub fn num_rows(&self) -> usize { self.data.len() }
  pub fn num_cols(&self) -> usize { self.data[0].len() }
}

impl<T, const rows: usize, const cols: usize> From<[[T; cols]; rows]> for Matrix<T, rows, cols> {
  fn from(data: [[T; cols]; rows]) -> Matrix<T, rows, cols> { Matrix{ data }}
}

impl<T: Clone + Copy + Default, const rows: usize> From<Tuple<T, rows>> for Matrix<T, rows, 1> {
  fn from(data: Tuple<T, rows>) -> Matrix<T, rows, 1> {
    let mut result = Matrix::<T, rows, 1>::with_dimensions().unwrap();
    for (i, val) in data.data.iter().enumerate() {
      result[i][0] = val.clone();
    }

    result
  }
}

impl<T: Default + Clone + Copy, const rows: usize, const cols: usize> Matrix<T, rows, cols> {
  pub fn new() -> Matrix<T, rows, cols> { Matrix{ data: [[T::default(); cols]; rows]}}
  
  pub fn with_dimensions<const drows: usize, const dcols: usize>() -> Result<Matrix<T, drows, dcols>, String> {
    if drows == 0 {
      return Err("can't create a matrix with 0 rows".to_string());
    }

    Ok(Matrix::from(
      [[T::default(); dcols]; drows]
    ))
  }
}

impl<T, const rows: usize, const cols: usize> Index<usize> for Matrix<T, rows, cols> {
  type Output = [T; cols];

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<T, const rows: usize, const cols: usize> IndexMut<usize> for Matrix<T, rows, cols> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut [T; cols] { &mut self.data[index] }
}

impl<T: Addable<T> + Copy, const rows: usize, const cols: usize> Add for &Matrix<T, rows, cols> {
  type Output = Option<Matrix<T, rows, cols>>;

  fn add(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None;
    }

    let mut matrix = Matrix::with_dimensions::<rows, cols>().unwrap();

    for row in 0..matrix.num_rows() {
      for col in 0..matrix.num_cols() {
        matrix[row][col] = self[row][col] + other[row][col];
      }
    }

    Some(matrix)
  }
}

impl<T: Subtractable<T> + Copy, const rows: usize, const cols: usize> Sub for &Matrix<T, rows, cols> {
  type Output = Option<Matrix<T, rows, cols>>;

  fn sub(self, other: Self) -> Self::Output {
    if self.data.len() != other.data.len() {
      return None;
    }

    let mut matrix: Matrix<T, rows, cols> = Matrix::with_dimensions::<rows, cols>().unwrap();

    for row in 0..matrix.num_rows() {
      for col in 0..matrix.num_cols() {
        matrix[row][col] = self[row][col] - other[row][col];
      }
    }

    Some(matrix)
  }
}

impl<const rows: usize, const cols: usize, const other_cols: usize> Mul<&Matrix<f64, cols, other_cols>> for &Matrix<f64, rows, cols> {
  type Output = Result<Matrix<f64, rows, other_cols>, String>;

  fn mul(self, other: &Matrix<f64, cols, other_cols>) -> Self::Output {
    let (rows, cols, shared) = (self.num_rows(), other.num_cols(), self.num_cols());
    if shared != other.num_rows() {
      return Err("the number of columns in the len matrix must be the same as the number of rows in the right matrix".to_string());
    }

    let mut result = Matrix::with_dimensions()?;

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

impl<const rows: usize, const cols: usize> Matrix<f64, rows, cols> {
  pub fn identity() -> Result<Self, String> {
    let mut result = Self::with_dimensions::<rows, cols>()?;

    for i in 0..rows {
      result[i][i] = 1.0;
    }

    Ok(result)
  }

  pub fn transpose<const trows: usize, const tcols: usize>(&self) -> Matrix<f64, trows, tcols> {
    let mut result = Self::with_dimensions::<trows, tcols>().unwrap();

    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[col][row] = self[row][col].clone();
      }
    }

    result
  }
}

impl<const rows: usize, const cols: usize> Matrix<f64, rows, cols> {
  pub fn scalar_mul(&self, scalar: f64) -> Matrix<f64, rows, cols> {
    let mut result = self.clone();
    
    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[row][col] *= scalar;
      }
    }

    result
  }

  pub fn scalar_div(&self, scalar: f64) -> Matrix<f64, rows, cols> {
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
        if upper_triangular[row][col] == 0.0 || upper_triangular[col][col] == 0.0 {
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

  pub fn submatrix<const subrows: usize, const subcols: usize>(&self, row: usize, col: usize) -> Result<Matrix<f64, subrows, subcols>, String> {
    if row >= self.num_rows() || col >= self.num_cols() {
      return Err(format!("submatrix does not exist for row={}, col={}", row, col));
    }

    let mut submatrix = Matrix::<f64, subrows, subcols>::with_dimensions().unwrap();
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

    let mut result = self.transpose::<rows, cols>();
    for row in 0..self.num_rows() {
      for col in 0..self.num_cols() {
        result[col][row] = self.cofactor(row, col)? / determinant;
      }
    }

    Ok(result)
  }
}

impl Matrix<f64, 4, 4> {
  pub fn translating(x: f64, y: f64, z: f64) -> Self {
    Self::from([
      // [1.0, 0.0, 0.0, x],
      // [0.0, 1.0, 0.0, y],
      // [0.0, 0.0, 1.0, z],
      // [0.0, 0.0, 0.0, 1.0],
      [1.0, 0.0,   x, 0.0],
      [0.0, 1.0,   y, 0.0],
      [0.0, 0.0,   z, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn scaling(sx: f64, sy: f64, sz: f64) -> Self {
    Self::from([
      [ sx, 0.0, 0.0, 0.0],
      [0.0,  sy, 0.0, 0.0],
      [0.0, 0.0,  sz, 0.0],
      [0.0, 0.0, 0.0, 1.0],
    ])
  }

  pub fn rotation_x(r: f64) -> Self {
    Self::from([
      [1.0,     0.0,      0.0, 0.0],
      [0.0, r.cos(), -r.sin(), 0.0],
      [0.0, r.sin(),  r.cos(), 0.0],
      [0.0,     0.0,      0.0, 1.0],
    ])
  }

  pub fn rotation_y(r: f64) -> Self {
    Self::from([
      [r.cos(),  0.0, r.sin(), 0.0],
      [0.0,      1.0,     0.0, 0.0],
      [-r.sin(), 0.0, r.cos(), 0.0],
      [0.0,      0.0,     0.0, 1.0],
    ])
  }

  pub fn rotation_z(r: f64) -> Self {
    Self::from([
      [r.cos(), -r.sin(), 0.0, 0.0],
      [r.sin(),  r.cos(), 0.0, 0.0],
      [0.0,          0.0, 1.0, 0.0],
      [0.0,          0.0, 0.0, 1.0],
    ])
  }

  pub fn shearing(xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
    Self::from([
      [1.0,  xy,  xz, 0.0],
      [ yx, 1.0,  yz, 0.0],
      [ zx,  zy, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
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
    const ROWS: usize = 2;
    const COLS: usize = 3;
    let result = Matrix::<i32, ROWS, COLS>::with_dimensions();

    let matrix = match result {
      Ok(val) => val,
      Err(why) => { panic!("{}", why);}
    };
    
    for row in 0..ROWS {
      for col in 0..COLS {
        assert_eq!(matrix[row][col], 0);
      }
    }
  }

  #[test]
  fn test_add() {
    let mat1 = Matrix::from([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    let mat2 = Matrix::from([
      [9, 8, 7],
      [6, 5, 4],
      [3, 2, 1],
    ]);
    let result = &mat1 + &mat2;
    let sum = match result {
      Some(val) => val,
      None => panic!()
    };
    let expected = Matrix::from([
      [10, 10, 10],
      [10, 10, 10],
      [10, 10, 10],
    ]);

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let mat1 = Matrix::from([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    let mat2 = Matrix::from([
      [9, 8, 7],
      [6, 5, 4],
      [3, 2, 1],
    ]);
    let result = &mat1 - &mat2;
    let diff = match result {
      Some(val) => val,
      None => panic!()
    };
    let expected = Matrix::from([
      [-8, -6, -4],
      [-2, 0, 2],
      [4, 6, 8],
    ]);

    assert_eq!(diff, expected);
  }

  #[test]
  fn test_mul() {
    let mat1 = Matrix::from([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 8.0, 7.0, 6.0],
      [5.0, 4.0, 3.0, 2.0],
    ]);
    let mat2 = Matrix::from([
      [-2.0, 1.0, 2.0,  3.0],
      [ 3.0, 2.0, 1.0, -1.0],
      [ 4.0, 3.0, 6.0,  5.0],
      [ 1.0, 2.0, 7.0,  8.0],
    ]);
    let expected = Matrix::from([
      [20.0, 22.0,  50.0,  48.0],
      [44.0, 54.0, 114.0, 108.0],
      [40.0, 58.0, 110.0, 102.0],
      [16.0, 26.0,  46.0,  42.0],
    ]);
    let mul = match &mat1 * &mat2 {
      Err(why) => panic!("{}", why),
      Ok(val) => val
    };

    assert_eq!(expected, mul);
  }

  #[test]
  fn test_tuple_mul() {
    let mat = Matrix::from([
      [1.0, 2.0, 3.0, 4.0],
      [2.0, 4.0, 4.0, 2.0],
      [8.0, 6.0, 4.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
    ]);
    let tuple = Tuple::from([1.0, 2.0, 3.0, 1.0]);
    let expected = Matrix::from(Tuple::from([18.0, 24.0, 33.0, 1.0]));
    let product = (&mat * &Matrix::<f64, 4, 1>::from(tuple)).unwrap();
  
    assert_eq!(expected, product);
  }

  #[test]
  fn test_transpose() {
    let mat = Matrix::from([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    let expected = Matrix::from([
      [1.0, 4.0, 7.0],
      [2.0, 5.0, 8.0],
      [3.0, 6.0, 9.0],
    ]);
    let transpose = mat.transpose();

    assert_eq!(transpose, expected);

    let mat = Matrix::from([
      [1.0, 2.0, 3.0, 1.0],
      [4.0, 5.0, 6.0, 2.0],
      [7.0, 8.0, 9.0, 3.0],
    ]);
    let expected = Matrix::from([
      [1.0, 4.0, 7.0],
      [2.0, 5.0, 8.0],
      [3.0, 6.0, 9.0],
      [1.0, 2.0, 3.0],
    ]);
    let transpose = mat.transpose();

    assert_eq!(transpose, expected);
  }

  #[test]
  fn test_scalar_mul() {
    let mat = Matrix::from([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    let product = mat.scalar_mul(2.0);
    let expected = Matrix::from([
      [2.0, 4.0],
      [6.0, 8.0],
    ]);

    assert_eq!(product, expected);
  }

  #[test]
  fn test_determinant() {
    let mat = Matrix::from([
      [-2.0, -8.0,  3.0,  5.0],
      [-3.0,  1.0,  7.0,  3.0],
      [ 1.0,  2.0, -9.0,  6.0],
      [-6.0,  7.0,  7.0, -9.0],
    ]);
    let determinant = mat.determinant().unwrap();
    let expected = -4071 as f64;

    assert_eq!(determinant, expected);

    let mat = Matrix::from([
      [ 1.0,  2.0,  6.0],
      [-5.0,  8.0, -4.0],
      [ 2.0,  6.0,  4.0],
    ]);
    let determinant = mat.determinant().unwrap();
    let expected = -196 as f64;

    assert_eq!(determinant, expected);
  }

  #[test]
  fn test_inverse() {
    let mat = Matrix::<f64, 4, 4>::from([
      [-5.0,  2.0,  6.0, -8.0],
      [ 1.0, -5.0,  1.0,  8.0],
      [ 7.0,  7.0, -6.0, -7.0],
      [ 1.0, -3.0,  7.0,  4.0],
    ]);
    let expected = Matrix::<f64, 4, 4>::from([
      [ 0.21805,  0.45113,  0.24060, -0.04511],
      [-0.80827, -1.45677, -0.44361,  0.52068],
      [-0.07895, -0.22368, -0.05263,  0.19737],
      [-0.52256, -0.81391, -0.30075,  0.30639],
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
