use std::{fs::File, path::Path};

use canvas::{Canvas, Color};
use lalg::{tuple::Coordinates, vector::Vector};
use raytracer::{ray::{Intersects, Ray}, sphere::Sphere};

pub mod lalg;
pub mod canvas;
pub mod raytracer;

fn main() {
    let mut canvas = Canvas::new(300, 300);

    let sphere = Sphere::new();
    let ray_origin = Vector::from(vec![0.0, 0.0, -5.0, 0.0]);
    let screen_position = Vector::from(vec![0.0, 0.0, -1.5, 0.0]);
    
    let canvas_dim_magnitude = (canvas.width().pow(2) as f64 + canvas.height().pow(2) as f64).sqrt();

    for x in 0..canvas.width() {
        for y in 0..canvas.height() {
            let pixel_position = Vector::from(
                vec![
                    (x as f64 - (canvas.width() / 2) as f64) / canvas_dim_magnitude,
                    (y as f64 - (canvas.height() / 2) as f64) / canvas_dim_magnitude,
                    *screen_position.z(),
                    0.0
                ]
            );
            let ray = Ray{ origin: ray_origin.clone(), direction: pixel_position.norm()};
            if sphere.intersections(&ray).len() > 0 {
                canvas[x][y] = Color::from(vec![1.0, 0.8, 0.5, 0.0]);
            } else {
                canvas[x][y] = Color::from(vec![0.3, 0.5, 1.0, 0.0]);
            }
        }
    }

    let mut file = File::create(Path::new("canvas.ppm")).unwrap();
    canvas.save_to_file(&mut file).unwrap();
}
