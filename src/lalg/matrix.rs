use std::ops::{Add, Index, IndexMut, Mul, Sub};

use super::{tuple::Tuple, util::cofactor};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<const ROWS: usize, const COLS: usize> {
  pub data: [[f64; COLS]; ROWS]
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
  pub fn num_rows(&self) -> usize { self.data.len() }
  pub fn num_cols(&self) -> usize { self.data[0].len() }
}

impl<const ROWS: usize, const COLS: usize> From<[[f64; COLS]; ROWS]> for Matrix<ROWS, COLS> {
  fn from(data: [[f64; COLS]; ROWS]) -> Matrix<ROWS, COLS> { Matrix{ data }}
}

impl<const ROWS: usize> From<Tuple<ROWS>> for Matrix<ROWS, 1> {
  fn from(data: Tuple<ROWS>) -> Matrix<ROWS, 1> {
    let mut result = Matrix::<ROWS, 1>::with_dimensions().unwrap();
    for (i, val) in data.data.iter().enumerate() {
      result[i][0] = val.clone();
    }

    result
  }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
  pub fn new() -> Matrix<ROWS, COLS> { Matrix{ data: [[f64::default(); COLS]; ROWS]}}
  
  pub fn with_dimensions<const DROWS: usize, const DCOLS: usize>() -> Result<Matrix<DROWS, DCOLS>, String> {
    if DROWS == 0 {
      return Err("can't create a matrix with 0 rows".to_string());
    }

    Ok(Matrix::from(
      [[f64::default(); DCOLS]; DROWS]
    ))
  }
}

impl<const ROWS: usize, const COLS: usize> Index<usize> for Matrix<ROWS, COLS> {
  type Output = [f64; COLS];

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<const ROWS: usize, const COLS: usize> IndexMut<usize> for Matrix<ROWS, COLS> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut [f64; COLS] { &mut self.data[index] }
}

impl<const ROWS: usize, const COLS: usize> Add for &Matrix<ROWS, COLS> {
  type Output = Matrix<ROWS, COLS>;

  fn add(self, other: Self) -> Self::Output {
    let mut matrix = Matrix::<ROWS, COLS>::with_dimensions::<ROWS, COLS>().unwrap();

    for row in 0..ROWS {
      for col in 0..COLS {
        matrix[row][col] = self[row][col] + other[row][col];
      }
    }

    matrix
  }
}

impl<const ROWS: usize, const COLS: usize> Sub for &Matrix<ROWS, COLS> {
  type Output = Matrix<ROWS, COLS>;

  fn sub(self, other: Self) -> Self::Output {
    let mut matrix: Matrix<ROWS, COLS> = Matrix::<ROWS, COLS>::with_dimensions::<ROWS, COLS>().unwrap();

    for row in 0..ROWS {
      for col in 0..COLS {
        matrix[row][col] = self[row][col] - other[row][col];
      }
    }

    matrix
  }
}

impl<const ROWS: usize, const COLS: usize, const OTHER_COLS: usize> Mul<&Matrix<COLS, OTHER_COLS>> for &Matrix<ROWS, COLS> {
  type Output = Matrix<ROWS, OTHER_COLS>;

  fn mul(self, other: &Matrix<COLS, OTHER_COLS>) -> Self::Output {
    let mut result = Matrix::<ROWS, OTHER_COLS>::with_dimensions().unwrap();

    for m in 0..ROWS {
      for n in 0..OTHER_COLS {
        for k in 0..COLS {
          result[m][n] += self[m][k] * other[k][n];
        }
      }
    }

    result
  }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
  pub fn identity() -> Result<Self, String> {
    let mut result = Self::with_dimensions::<ROWS, COLS>()?;

    for i in 0..ROWS {
      result[i][i] = 1.0;
    }

    Ok(result)
  }

  pub fn transpose<const TROWS: usize, const TCOLS: usize>(&self) -> Matrix<TROWS, TCOLS> {
    let mut result = Self::with_dimensions::<TROWS, TCOLS>().unwrap();

    for row in 0..ROWS {
      for col in 0..COLS {
        result[col][row] = self[row][col].clone();
      }
    }

    result
  }
}

impl<const ROWS: usize, const COLS: usize> Matrix<ROWS, COLS> {
  pub fn scalar_mul(&self, scalar: f64) -> Matrix<ROWS, COLS> {
    let mut result = self.clone();
    
    for row in 0..ROWS {
      for col in 0..COLS {
        result[row][col] *= scalar;
      }
    }

    result
  }

  pub fn scalar_div(&self, scalar: f64) -> Matrix<ROWS, COLS> {
    let mut result = self.clone();
    
    for row in 0..ROWS {
      for col in 0..COLS {
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
    if COLS != ROWS {
      return Err("cannot get the determinant of a non-square matrix".to_string());
    }

    let mut upper_triangular = self.clone();
    // ensure the diagonal has non-zero values
    for i in 0..ROWS {
      if upper_triangular[i][i] != 0.0 {
        continue;
      }
      for j in i+1..ROWS {
        if upper_triangular[j][i] == 0.0 {
          continue;
        }
        upper_triangular.data.swap(i, j);
        break;
      }
    }

    for row in 1..ROWS {
      for col in 0..row {
        // avoid divide-by-zero errors
        if upper_triangular[row][col] == 0.0 || upper_triangular[col][col] == 0.0 {
          continue;
        }
        let factor = upper_triangular[row][col] / upper_triangular[col][col];
        for k in 0..COLS {
          upper_triangular[row][k] -= factor * upper_triangular[col][k];
        }
      }
    }

    let mut det = 1 as f64;
    for i in 0..ROWS {
      det *= upper_triangular[i][i];
    }

    Ok(det)
  }

  pub fn inverse(&self) -> Result<Self, String> {
    let determinant = self.determinant()?;
    if determinant == 0.0 {
      return Err("cannot invert a matrix if it's determinant is zero".to_string());
    }

    let mut result = self.transpose::<ROWS, COLS>();    
    let matrix = self.data.iter().map(|row| Vec::from(row.clone())).collect::<Vec<Vec<f64>>>();
    for row in 0..ROWS {
      for col in 0..COLS {
        result[col][row] = cofactor(&matrix, row, col)? / determinant;
      }
    }

    Ok(result)
  }
}

impl Matrix<4, 4> {
  pub fn translating(x: f64, y: f64, z: f64) -> Self {
    Self::from([
      [1.0, 0.0, 0.0,   x],
      [0.0, 1.0, 0.0,   y],
      [0.0, 0.0, 1.0,   z],
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
      [0.0, r.cos(), -(r.sin()), 0.0],
      [0.0, r.sin(),  r.cos(), 0.0],
      [0.0,     0.0,      0.0, 1.0],
    ])
  }

  pub fn rotation_y(r: f64) -> Self {
    Self::from([
      [ f64::cos(r), 0.0, f64::sin(r), 0.0],
      [         0.0, 1.0,         0.0, 0.0],
      [-f64::sin(r), 0.0, f64::cos(r), 0.0],
      [         0.0, 0.0,         0.0, 1.0],
    ])
  }

  pub fn rotation_z(r: f64) -> Self {
    Self::from([
      [r.cos(), -(r.sin()), 0.0, 0.0],
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
    &Self::translating(x, y, z) * self
  }

  pub fn scale(&self, x: f64, y: f64, z: f64) -> Self {
    &Self::scaling(x, y, z) * self
  }

  pub fn rotate_x(&self, r: f64) -> Self {
    &Self::rotation_x(r) * self
  }

  pub fn rotate_y(&self, r: f64) -> Self {
    &Self::rotation_y(r) * self
  }

  pub fn rotate_z(&self, r: f64) -> Self {
    &Self::rotation_z(r) * self
  }

  pub fn shear(&self, xy: f64, xz: f64, yx: f64, yz: f64, zx: f64, zy: f64) -> Self {
    &Self::shearing(xy, xz, yx, yz, zx, zy) * self
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_with_dimensions() {
    const ROWS: usize = 2;
    const COLS: usize = 3;
    let result = Matrix::<ROWS, COLS>::with_dimensions::<ROWS, COLS>();

    let matrix = match result {
      Ok(val) => val,
      Err(why) => { panic!("{}", why);}
    };
    
    for row in 0..ROWS {
      for col in 0..COLS {
        assert_eq!(matrix[row][col], 0.0);
      }
    }
  }

  #[test]
  fn test_add() {
    let mat1 = Matrix::from([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    let mat2 = Matrix::from([
      [9.0, 8.0, 7.0],
      [6.0, 5.0, 4.0],
      [3.0, 2.0, 1.0],
    ]);
    let sum = &mat1 + &mat2;
    let expected = Matrix::from([
      [10.0, 10.0, 10.0],
      [10.0, 10.0, 10.0],
      [10.0, 10.0, 10.0],
    ]);

    assert_eq!(sum, expected);
  }

  #[test]
  fn test_sub() {
    let mat1 = Matrix::from([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0],
    ]);
    let mat2 = Matrix::from([
      [9.0, 8.0, 7.0],
      [6.0, 5.0, 4.0],
      [3.0, 2.0, 1.0],
    ]);
    let diff = &mat1 - &mat2;
    let expected = Matrix::from([
      [-8.0, -6.0, -4.0],
      [-2.0,  0.0,  2.0],
      [ 4.0,  6.0,  8.0],
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
    let mul = &mat1 * &mat2;

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
    let product = &mat * &Matrix::from(tuple);
  
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
    let mat = Matrix::from([
      [-5.0,  2.0,  6.0, -8.0],
      [ 1.0, -5.0,  1.0,  8.0],
      [ 7.0,  7.0, -6.0, -7.0],
      [ 1.0, -3.0,  7.0,  4.0],
    ]);
    let expected = Matrix::from([
      [ 0.21805,  0.45113,  0.24060, -0.04511],
      [-0.80827, -1.45677, -0.44361,  0.52068],
      [-0.07895, -0.22368, -0.05263,  0.19737],
      [-0.52256, -0.81391, -0.30075,  0.30639],
    ]);
    let inverse = mat.inverse().unwrap();
    let diff = &inverse - &expected;
    for row in 0..inverse.num_rows() {
      for col in 0..inverse.num_cols() {
        assert!(diff[row][col].abs() < 1e-5)
      }
    }

    let mat = Matrix::from([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]);
    let expected = Matrix::from([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0],
      [0.0, 0.0, 0.0, 1.0]
    ]);
    let inverse = mat.inverse().unwrap();
    let diff = &inverse - &expected;
    for row in 0..inverse.num_rows() {
      for col in 0..inverse.num_cols() {
        assert!(diff[row][col].abs() < 1e-5)
      }
    }
  }

  #[test]
  fn test_rotate_y() {
    let mat = Matrix::rotation_y(2.0);
    assert_eq!(mat, Matrix::from([
      [(2.0 as f64).cos(),  0.0, (2.0 as f64).sin(), 0.0],
      [               0.0,  1.0,                0.0, 0.0],
      [-(2.0 as f64).sin(), 0.0, (2.0 as f64).cos(), 0.0],
      [                0.0, 0.0,                0.0, 1.0],
    ]));
    
    let inv = mat.inverse().unwrap();
    assert_eq!(inv, Matrix::from([
      [-0.4161468365471424, 0.0,  0.9092974268256817, 0.0],
      [ 0.0,                1.0,                 0.0, 0.0],
      [-0.9092974268256817, 0.0, -0.4161468365471424, 0.0],
      [ 0.0,                0.0,                 0.0, 1.0],
    ]));

    // assert!(false);
  }
}
