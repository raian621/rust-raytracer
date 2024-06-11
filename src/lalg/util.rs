pub fn submatrix(
  matrix: &Vec<Vec<f64>>,
  row: usize,
  col: usize
) -> Result<Vec<Vec<f64>>, String> {
  let rows = matrix.len();
  let cols = matrix[0].len();

  if row >= rows || col >= cols {
    return Err(format!("cannot make a submatrix for row {} and col {} from matrix with {} rows and {} cols", row, col, rows, cols));
  }

  let mut submat = vec![vec![0.0; matrix[0].len()-1]; matrix.len()-1];
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


pub fn cofactor(matrix: &Vec<Vec<f64>>, row: usize, col: usize) -> Result<f64, String> {
  match (row + col) % 2 {
    0 => Ok(minor(matrix, row, col)?),
    _ => Ok(-minor(matrix, row, col)?)
    }
}
pub fn minor(matrix: &Vec<Vec<f64>>, row: usize, col: usize) -> Result<f64, String> {
  let submat = submatrix(matrix, row, col)?;
  Ok(determinant(&submat)?)
}

pub fn determinant(matrix: &Vec<Vec<f64>>) -> Result<f64, String> {
  let rows = matrix.len();
  let cols = matrix[0].len();
  if rows != cols {
    return Err("cannot get the determinant of a non-square matrix".to_string());
  }

  let mut upper_triangular = matrix.clone();
  // ensure the diagonal has non-zero values
  for i in 0..rows {
    if upper_triangular[i][i] != 0.0 {
      continue;
    }
    for j in i+1..rows {
      if upper_triangular[j][i] == 0.0 {
        continue;
      }
      upper_triangular.swap(i, j);
      break;
    }
  }

  for row in 1..rows {
    for col in 0..row {
      // avoid divide-by-zero errors
      if upper_triangular[row][col] == 0.0 || upper_triangular[col][col] == 0.0 {
        continue;
      }
      let factor = upper_triangular[row][col] / upper_triangular[col][col];
      for k in 0..cols {
        upper_triangular[row][k] -= factor * upper_triangular[col][k];
      }
    }
  }

  let mut det = 1.0;
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
