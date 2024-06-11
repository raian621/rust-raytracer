use std::ops::{Add, Div, Mul, Sub};

#[derive(Clone, Debug, PartialEq)]
pub struct Color {
  pub r: f64,
  pub g: f64,
  pub b: f64,
  pub a: f64
}

impl Default for Color {
  fn default() -> Self {
    Self{ r: 1.0, g: 1.0, b: 1.0, a: 0.0 }
  }
}

impl Add<Self> for &Color {
  type Output = Color;

  fn add(self, other: Self) -> Self::Output {
    Color{ 
      r: self.r + other.r,
      g: self.g + other.g,
      b: self.b + other.b,
      a: self.a + other.a
    }
  }
}

impl Sub<Self> for &Color {
  type Output = Color;

  fn sub(self, other: Self) -> Self::Output {
    Color{ 
      r: self.r - other.r,
      g: self.g - other.g,
      b: self.b - other.b,
      a: self.a - other.a
    }
  }
}

impl Mul<Self> for &Color {
  type Output = Color;

  fn mul(self, other: Self) -> Self::Output {
    Color{ 
      r: self.r * other.r,
      g: self.g * other.g,
      b: self.b * other.b,
      a: self.a * other.a
    }
  }
}

impl Div<Self> for &Color {
  type Output = Color;

  fn div(self, other: Self) -> Self::Output {
    Color{ 
      r: self.r / other.r,
      g: self.g / other.g,
      b: self.b / other.b,
      a: self.a / other.a
    }
  }
}

impl Color {
  pub fn color(r: f64, g: f64, b: f64) -> Self { Self{ r, g, b, a: 1.0 }}

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