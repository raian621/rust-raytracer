use std::{fs::File, io::{self, Write}, ops::{Index, IndexMut}};

use crate::lalg::vector::Vector;

pub type Color<const SIZE: usize> = Vector<SIZE>;
pub struct Canvas<const WIDTH: usize, const HEIGHT: usize> { 
  pub data: [[Color<3>; HEIGHT]; WIDTH]
}

impl Color<3> {
  pub fn r(&self) -> f64 { self[0] }
  pub fn g(&self) -> f64 { self[1] }
  pub fn b(&self) -> f64 { self[2] }
}

impl Color<4> {
  pub fn r(&self) -> f64 { self[0] }
  pub fn g(&self) -> f64 { self[1] }
  pub fn b(&self) -> f64 { self[2] }
  pub fn a(&self) -> f64 { self[3] }
}

impl<const WIDTH: usize, const HEIGHT: usize> Canvas<WIDTH, HEIGHT> {
  pub fn width(&self) ->  usize { WIDTH }
  pub fn height(&self) -> usize { HEIGHT }

  pub fn new() -> Self { 
    Self { 
      data: [[Color::from([0.0, 0.0, 0.0]); HEIGHT]; WIDTH]
    }
  }

  pub fn save_to_file(&self, file: &mut File) -> io::Result<usize> {
    let mut bytes_written = 0;

    bytes_written += self.write_header(file)?;
    // keep lines under 70 characters long
    let mut line = Vec::<String>::new();
    let max_line_length = 80;
    let mut line_length = 0;
    for x in 0..WIDTH {
      for y in 0..HEIGHT {
        let red = ((self[x][y].r() * 256.0) as u8).to_string();
        let green = ((self[x][y].g() * 256.0) as u8).to_string();
        let blue = ((self[x][y].b() * 256.0) as u8).to_string();

        for val in vec![red, green, blue].into_iter() {
          if line_length == 0 {
            line_length += val.len();
            line.push(val);
            continue;
          }

          if line_length + val.len() + 1 > max_line_length {
            bytes_written += file.write(format!("{}\n", line.join(" ")).as_bytes())?;
            line_length = val.len();
            line = vec![val];
          } else {
            line_length += val.len() + 1;
            line.push(val);
          }
        }
      }
    }

    if line.len() > 1 {
      bytes_written += file.write(format!("{}\n", line.join(" ")).as_bytes())?;
    }
    Ok(bytes_written)
  }

  fn write_header(&self, file: &mut File) -> io::Result<usize> {
    file.write(format!("P3\n{} {}\n255\n", WIDTH, HEIGHT).as_bytes())
  }
}

impl<const WIDTH: usize, const HEIGHT: usize> Index<usize> for Canvas<WIDTH, HEIGHT> {
  type Output = [Color<3>; HEIGHT];

  fn index(&self, index: usize) -> &Self::Output { &self.data[index] }
}

impl<const WIDTH: usize, const HEIGHT: usize> IndexMut<usize> for Canvas<WIDTH, HEIGHT> {
  fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut [Color<3>; HEIGHT] { &mut self.data[index] }
}

#[cfg(test)]
pub mod tests {
  use super::*;

  #[test]
  fn test_color_init() {
    let (r, g, b, a) = (0.2, 0.5, 0.7, 0.9);
    let color = Color::from([r, g, b]);
    assert_eq!(color.r(), r);
    assert_eq!(color.g(), g);
    assert_eq!(color.b(), b);

    let color = Color::from([r, g, b, a]);
    assert_eq!(color.r(), r);
    assert_eq!(color.g(), g);
    assert_eq!(color.b(), b);
    assert_eq!(color.a(), a);
  }
}