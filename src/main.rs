use std::{fs::File, path::Path};

use lalg::{matrix::Matrix, vector::Vector, tuple::Coordinates};
use raytracer::{canvas::Canvas, point_light::PointLight, ray::Ray};
use shapes::{color::Color, intersection::Intersects, sphere::Sphere};

pub mod lalg;
pub mod raytracer;
pub mod shapes;

fn main() {
  draw_sphere();
}

fn draw_sphere() {
  let mut canvas = Box::new(Canvas::<200, 200>::new());
  let mut sphere = Sphere::new();
  sphere.material.color = Color::magneta();
  // sphere.transform = &Matrix::rotation_y(1.2) * &Matrix::scaling(1.0, 1.0, 0.2);
  // sphere.transform = Matrix::scaling(0.5, 1.0, 1.0);

  let light = PointLight{
    intensity: Color::white(),
    position: Vector::from([-10.0, 10.0, -10.0, 1.0])
  };

  let ray_origin = Vector::from([0.0, 0.0, 10.0, 1.0]);

  let canvas_dim_magnitude =
    (canvas.width().pow(2) as f64 + canvas.height().pow(2) as f64).sqrt();

  for x in 0..canvas.width() {
    for y in 0..canvas.height() {
      let pixel_position = Vector::from([
        (x as f64 - (canvas.width() / 2) as f64) / canvas_dim_magnitude,
        (y as f64 - (canvas.height() / 2) as f64) / canvas_dim_magnitude,
        3.0,
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
        let eye = -&ray.direction;
        let color = sphere.material.lighting(&light, &intersection_point, &eye, &normal);
        canvas[x][y] = color;
      } else {       
        // canvas[x][y] = Color::from([0.3, 0.5, 1.0]);
        canvas[x][y] = Color::cyan();
      }

      print!("{}%\r", 100.0 * x as f64 / canvas.width() as f64);
    }
  }

  let mut file = File::create(Path::new("canvas.ppm")).unwrap();
  canvas.save_to_file(&mut file).unwrap();
}
