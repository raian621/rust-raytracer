use std::{fs::File, path::Path};

use canvas::{Canvas, Color};
use lalg::{tuple::Coordinates, vector::Vector};
use raytracer::{
    ray::{Intersects, Ray},
    sphere::Sphere,
};

pub mod canvas;
pub mod lalg;
pub mod raytracer;

fn main() {
  let mut canvas = Canvas::<1000, 1000>::new();
  let mut sphere: Sphere = Sphere::new();
  sphere.transform = sphere.transform.scale(4.0, 1.0, 1.0);
  let ray_origin = Vector::from([0.0, 0.0, -5.0, 0.0]);
  let screen_position = Vector::from([0.0, 0.0, -1.5, 0.0]);

  let canvas_dim_magnitude =
    (canvas.width().pow(2) as f64 + canvas.height().pow(2) as f64).sqrt();

  for x in 0..canvas.width() {
    for y in 0..canvas.height() {
      let pixel_position = Vector::from([
        (x as f64 - (canvas.width() / 2) as f64) / canvas_dim_magnitude,
        (y as f64 - (canvas.height() / 2) as f64) / canvas_dim_magnitude,
        *screen_position.z(),
        0.0,
      ]);
      let ray = Ray {
        origin: ray_origin.clone(),
        direction: pixel_position.norm(),
      };

      let intersections = sphere.intersections(&ray);

      if intersections.len() > 0 {
        let intersection = &intersections[0];
        let intersection_point = ray.get_point(intersection.t);
        let normal = sphere.normal_at(&intersection_point);
        canvas[x][y] = Color::from([*normal.x(), *normal.y(), *normal.z()]);
      } else {
        canvas[x][y] = Color::from([0.3, 0.5, 1.0]);
      }

      // print!("{}%\r", 100.0 * x as f64 / canvas.width() as f64);
    }
  }

  let mut file = File::create(Path::new("canvas.ppm")).unwrap();
  canvas.save_to_file(&mut file).unwrap();
}
