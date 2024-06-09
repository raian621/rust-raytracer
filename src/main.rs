use std::{fs::File, path::Path};

use canvas::{Canvas, Color};
use lalg::{matrix::Matrix, tuple::Coordinates, vector::Vector};

pub mod lalg;
pub mod canvas;
pub mod raytracer;

fn main() {
    let mut canvas = Canvas::new(120, 120);

    let vec = Vector::<f64>::from(vec![50.0, 0.0, 0.0, 1.0]);

    let points = 1200 as usize;
    for i in 0..points {
        let pos = Matrix::from(vec.clone())
            .rotate_z(2.0 * std::f64::consts::PI * i as f64 / points as f64)
            .translate(canvas.width() as f64 / 2.0, canvas.height() as f64 / 2.0, 0.0)
            .transpose();

        let point = Vector::from(pos[0].clone());
        canvas[*point.x() as usize][*point.y() as usize] = Color::from(vec![1.0, 1.0, 1.0]);
    }

    let mut file = File::create(Path::new("canvas.ppm")).unwrap();
    canvas.save_to_file(&mut file).unwrap();
}
