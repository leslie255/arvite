use cgmath::*;

#[derive(Default, Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn into_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    pub const fn into_vec4(self) -> Vector4<f32> {
        vec4(self.r, self.g, self.b, self.a)
    }
}

impl From<Color> for [f32; 4] {
    fn from(color: Color) -> Self {
        color.into_array()
    }
}

impl From<Color> for Vector4<f32> {
    fn from(color: Color) -> Self {
        color.into_vec4()
    }
}


