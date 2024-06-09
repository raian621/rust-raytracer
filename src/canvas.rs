use std::{fs::File, io::{self, Write}};

use crate::lalg::{matrix::Matrix, vector::Vector};

pub type Color = Vector<f32>;
pub type Canvas = Matrix<Color>;

impl Color {
  pub fn r(&self) -> f32 { self[0] }
  pub fn g(&self) -> f32 { self[1] }
  pub fn b(&self) -> f32 { self[2] }
  pub fn a(&self) -> f32 { self[3] }
}

impl Canvas {
  pub fn width(&self) ->  usize { self.num_rows() }
  pub fn height(&self) -> usize { self.num_cols() }

  pub fn new(width: usize, height: usize) -> Self { 
    Self { 
      data: vec![vec![Color { data: vec![0 as f32; 3]}; height]; width]
    }
  }

  pub fn save_to_file(&self, file: &mut File) -> io::Result<usize> {
    let mut bytes_written = 0;

    bytes_written += self.write_header(file)?;
    // keep lines under 70 characters long
    let mut line = Vec::<String>::new();
    let max_line_length = 80;
    let mut line_length = 0;
    for x in 0..self.width() {
      for y in 0..self.height() {
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

    Ok(bytes_written)
  }

  fn write_header(&self, file: &mut File) -> io::Result<usize> {
    file.write(format!("P3\n{} {}\n255\n", self.width(), self.height()).as_bytes())
  }
}