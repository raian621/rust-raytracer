use std::str::SplitWhitespace;

use crate::{lalg::{tuple::Tuple, vector::Vector}, raytracer::point_light::PointLight};

use super::color::Color;

#[derive(Debug, Clone, PartialEq)]

pub struct Material {
  pub color: Color,
  pub ambient: f64,
  pub diffuse: f64,
  pub specular: f64,
  pub shininess: f64
}

impl Default for Material {
  fn default() -> Self {
    Self {
      color: Color::white(),
      ambient: 0.1,
      diffuse: 0.9,
      specular: 0.9,
      shininess: 200.0
    }
  }
}

impl Material {
  pub fn lighting(
    &self,
    light: &PointLight,
    point: &Vector<4>,
    eye_vector: &Vector<4>,
    normal_vector: &Vector<4>
  ) -> Color {
    let effective_color = &self.color * &light.intensity;
    // find the direction to the light source
    let light_vector = (&light.position - point).norm();
    let mut ambient = effective_color.clone();
    ambient.r *= self.ambient;
    ambient.g *= self.ambient;
    ambient.b *= self.ambient;

    let light_dot_normal = light_vector.dot(normal_vector);
    let mut diffuse = Color::black();
    let mut specular = Color::black();
    
    if light_dot_normal >= 0.0 {
      diffuse = effective_color;
      diffuse.r *= self.diffuse * light_dot_normal;
      diffuse.g *= self.diffuse * light_dot_normal;
      diffuse.b *= self.diffuse * light_dot_normal;

      let reflect_vector = (-&light_vector).reflect(&normal_vector);
      let reflect_dot_eye = reflect_vector.dot(eye_vector);
      
      if reflect_dot_eye > 0.0 {
        let factor = f64::powf(reflect_dot_eye, self.shininess);
        specular = light.intensity.clone();
        specular.r *= self.specular * factor;
        specular.g *= self.specular * factor;
        specular.b *= self.specular * factor;

      return &(&ambient + &diffuse) + &specular
       
      }
    }

    &(&ambient + &diffuse) + &specular
  }
}

#[cfg(test)]
pub mod tests {
  use super::*;

  #[test]
  fn test_lighting() {
    let material = Material::default();
    let position = Vector::from([0.0, 0.0, 0.0, 1.0]);
    let normal = Vector::from([0.0, 0.0, -1.0, 0.0]);
    let eye = Vector::from([0.0, 0.0, -1.0, 0.0]);

    let light = PointLight{
      intensity: Color::default(),
      position: Vector::from([0.0, 0.0, -10.0, 1.0])
    };

    let result = material.lighting(&light, &position, &eye, &normal);
    let expected = Color::color(1.9, 1.9, 1.9);

    assert_eq!(result, expected);

    let eye = Vector::from([0.0, f64::sqrt(2.0)/2.0, f64::sqrt(2.0)/2.0, 0.0]);
    let result = material.lighting(&light, &position, &eye, &normal);
    let expected = Color::color(1.0, 1.0, 1.0);

    assert_eq!(result, expected);
  }
}