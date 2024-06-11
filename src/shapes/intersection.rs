use crate::raytracer::ray::Ray;

#[derive(Debug, Clone, PartialEq)]
pub struct Intersection<'a, T> {
  pub t: f64,
  pub object: &'a T
}

pub trait Intersects<T> {
  fn intersections(&self, ray: &Ray) -> Vec<Intersection<T>>;
}