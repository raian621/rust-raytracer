use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Addable<T>: Default + Clone + Add<Output = T> {}
impl<T> Addable<T> for T where T: Default + Clone + Add<Output = T> {}

pub trait Subtractable<T>: Default + Clone + Sub<Output = T> {}
impl<T> Subtractable<T> for T where T: Default + Clone + Sub<Output = T> {}

pub trait Negateable<T>: Default + Clone + Neg<Output = T> {}
impl<T> Negateable<T> for T where T: Default + Clone + Neg<Output = T> {}

pub trait Multipliable<T>: Default + Clone + Copy + Mul<Output = T> {}
impl<T> Multipliable<T> for T where T: Default + Clone + Copy + Mul<Output = T> {}

pub trait Dividable<T>: Default + Clone + Copy + Div<Output = T> {}
impl<T> Dividable<T> for T where T: Default + Clone + Copy + Div<Output = T> {}
