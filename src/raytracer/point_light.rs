use crate::{lalg::vector::Vector, shapes::color::Color};

pub struct PointLight {
  pub intensity: Color,
  pub position: Vector<4>
}

impl PointLight {
  pub fn new() -> Self {
    PointLight{intensity: Color::default(), position: Vector::from([0.0, 0.0, 0.0, 1.0])}
  }
}