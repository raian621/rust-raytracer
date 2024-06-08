use std::{fs::File, path::Path};

use canvas::{Canvas, Color};

pub mod lalg;
pub mod canvas;

fn main() {
    let mut canvas = Canvas::new(120, 120);

    for x in 0..canvas.width() {
        canvas[x][60] = Color::from(vec![1.0, 1.0, 1.0]);
        canvas[60][x] = Color::from(vec![1.0, 1.0, 1.0]);
    }

    let mut file = match File::create(Path::new("canvas.ppm")) {
        Err(why) => panic!("{}", why),
        Ok(file) => file
    };

    canvas.save_to_file(&mut file).unwrap();
}
