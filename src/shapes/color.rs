use crate::lalg::vector::Vector;

pub type Color<const SIZE: usize> = Vector<SIZE>;

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

impl Default for Color<3> {
  fn default() -> Self {
    Self{ data: [1.0, 1.0, 1.0]}
  }
}

impl Color<3> {
  pub fn color(r: f64, g: f64, b: f64) -> Self { Self{ data: [r, g, b] }}

  pub fn red() -> Self     { Self::color(1.0, 0.0, 0.0)}
  pub fn yellow() -> Self  { Self::color(1.0, 1.0, 0.0)}
  pub fn green() -> Self   { Self::color(0.0, 1.0, 0.0)}
  pub fn cyan() -> Self    { Self::color(0.0, 1.0, 1.0)}
  pub fn blue() -> Self    { Self::color(0.0, 0.0, 1.0)}
  pub fn magneta() -> Self { Self::color(1.0, 0.0, 1.0)}
  pub fn white() -> Self   { Self::color(1.0, 1.0, 1.0)}
  pub fn black() -> Self   { Self::color(0.0, 0.0, 0.0)}
  pub fn grey() -> Self    { Self::color(0.5, 0.5, 0.5)}
}