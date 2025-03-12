use cgmath::*;

pub trait SplineAlgorithm {
    fn characteristic_matrix() -> Matrix4<f32>;
}
