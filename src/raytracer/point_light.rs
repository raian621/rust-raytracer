use crate::{lalg::vector::Vector, shapes::color::Color};

pub struct PointLight {
  pub intensity: Color<3>,
  pub position: Vector<4>
}

impl PointLight {
  pub fn new() -> Self {
    PointLight{intensity: Vector::from([1.0, 1.0, 1.0]), position: Vector::from([0.0, 0.0, 0.0, 1.0])}
  }
}