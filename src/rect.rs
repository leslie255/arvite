use cgmath::*;
use glium::Surface as _;

use crate::{
    application::Context,
    color::Color,
    mesh::{self, Mesh},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

glium::implement_vertex!(RectVertex, position, color);

#[derive(Debug, Clone, Copy)]
pub enum FillColor {
    Uniform(Color),
    GradientHorizontal(Color, Color),
    GradientVertical(Color, Color),
}

impl From<Color> for FillColor {
    fn from(v: Color) -> Self {
        Self::Uniform(v)
    }
}

impl Default for FillColor {
    fn default() -> Self {
        Color::default().into()
    }
}

#[derive(Debug)]
pub struct Rect<'cx> {
    mesh: Mesh<'cx, RectVertex>,
    fill_color: FillColor,
    size: Vector2<f32>,
}

impl<'cx> Rect<'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            mesh: Mesh::new(context),
            fill_color: FillColor::default(),
            size: vec2(0., 0.),
        }
    }

    pub fn context(&self) -> &'cx Context {
        self.mesh.context()
    }

    pub fn set_size(&mut self, new_size: Vector2<f32>) {
        self.size = new_size;
        self.rebuild_mesh_data();
    }

    pub fn size(&self) -> Vector2<f32> {
        self.size
    }

    pub fn set_fill_color(&mut self, new_fill_color: FillColor) {
        self.fill_color = new_fill_color;
        self.rebuild_mesh_data();
    }

    pub fn uniform_fill(&mut self, color: Color) {
        self.set_fill_color(color.into());
    }

    pub fn gradient_fill_horizontal(&mut self, color_left: Color, color_right: Color) {
        self.set_fill_color(FillColor::GradientHorizontal(color_left, color_right));
    }

    pub fn gradient_fill_vertical(&mut self, color_top: Color, color_bottom: Color) {
        self.set_fill_color(FillColor::GradientVertical(color_top, color_bottom));
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        match self.fill_color {
            FillColor::Uniform(color) if color.a.is_zero() => return,
            _ => (),
        }
        self.mesh.update_if_needed();
        let (frame_width, frame_height) = frame.get_dimensions();
        let model_view = Matrix4::from_translation(Vector3::new(position.x, position.y, 0.));
        let projection = cgmath::ortho(0., frame_width as f32, frame_height as f32, 0., -1., 1.);
        self.mesh.draw(
            frame,
            glium::uniform! {
                model_view: mesh::matrix4_to_array(model_view),
                projection: mesh::matrix4_to_array(projection),
            },
            &self.context().shader_rect,
            &mesh::default_2d_draw_parameters(),
        );
    }

    fn rebuild_mesh_data(&mut self) {
        let (color_top_left, color_top_right, color_bottom_left, color_bottom_right) =
            match self.fill_color {
                FillColor::Uniform(color) => (color, color, color, color),
                FillColor::GradientHorizontal(color0, color1) => (color0, color1, color0, color1),
                FillColor::GradientVertical(color0, color1) => (color0, color0, color1, color1),
            };
        let vertices_data = [
            RectVertex {
                position: [0., self.size.y],
                color: color_bottom_left.into_array(),
            },
            RectVertex {
                position: [self.size.x, 0.],
                color: color_top_right.into_array(),
            },
            RectVertex {
                position: [self.size.x, self.size.y],
                color: color_bottom_right.into_array(),
            },
            RectVertex {
                position: [0., 0.],
                color: color_top_left.into_array(),
            },
        ];
        #[rustfmt::skip]
        let indices_data = [
            0, 1, 2,
            1, 0, 3,
        ];
        let (vertices, indices) = self.mesh.vertices_indices_mut();
        vertices.clear();
        vertices.extend_from_slice(&vertices_data);
        indices.clear();
        indices.extend_from_slice(&indices_data);
    }
}
