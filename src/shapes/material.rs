#[derive(Debug, Clone, PartialEq)]

pub struct Material {
  pub ambient: f64,
  pub diffuse: f64,
  pub specular: f64,
  pub shininess: f64
}

impl Default for Material {
  fn default() -> Self {
    Self { ambient: 1.0, diffuse: 1.0, specular: 1.0, shininess: 100.0 }
  }
}