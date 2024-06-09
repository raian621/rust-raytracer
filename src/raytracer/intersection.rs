#[derive(Debug, Clone, PartialEq)]
pub struct Intersection<'a, T> {
  pub t: f64,
  pub object: &'a T
}