use std::{borrow::Cow, mem};

use cgmath::*;
use glium::{Surface as _, index::PrimitiveType};

use crate::{
    bezier,
    color::Color,
    context::Context,
    match_into_unchecked,
    mesh::{self, Mesh},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathDrawingMode {
    Line,
    Fill,
}

/// Vertex used for `Rect` and `Path`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ColoredVertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}

glium::implement_vertex!(ColoredVertex, position, color);

#[derive(Debug)]
pub struct BezierSplinePath<'a, 'cx> {
    path: Path<'cx>,
    points: Cow<'a, [[Point2<f32>; 3]]>,
    color: Color,
    resolution: u32,
    needs_rebuild: bool,
}

impl<'a, 'cx> BezierSplinePath<'a, 'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            path: Path::new(context),
            points: Cow::default(),
            color: Color::default(),
            resolution: 48,
            needs_rebuild: false,
        }
    }

    pub fn segments<'b>(&'b self) -> &'b [[Point2<f32>; 3]]
    where
        'a: 'b,
    {
        &self.points
    }

    pub fn segments_mut(&mut self) -> &mut Vec<[Point2<f32>; 3]> {
        self.needs_rebuild = true;
        let points = mem::take(&mut self.points);
        self.points = points.into_owned().into();
        unsafe { match_into_unchecked!(&mut self.points, Cow::Owned(vec) => vec) }
    }

    pub fn set_resolution(&mut self, resolution: u32) {
        self.needs_rebuild = true;
        self.resolution = resolution;
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn set_color(&mut self, color: Color) {
        self.color = color;
    }

    pub fn color(&self) -> Color {
        self.color
    }

    fn rebuild_mesh_if_needed(&mut self) {
        if !self.needs_rebuild {
            return;
        }
        self.needs_rebuild = false;
        self.path.clear();
        if self.segments().len() <= 1 {
            return;
        }
        let context = self.context();
        let mut path = mem::replace(&mut self.path, Path::new(context));
        if self.segments().len() <= 1 {
            return;
        }
        for (i, segment) in self.segments().iter().enumerate() {
            let next_segment = match self.segments().get(i + 1) {
                Some(xs) => xs,
                None => &self.segments()[0],
            };
            let points = [segment[1], segment[2], next_segment[0], next_segment[1]];
            if points[1] == points[0] && points[2] == points[3] {
                // Special optimization for when having a segment of just straight lines.
                path.push_point(points[0], self.color);
                path.push_point(points[3], self.color);
                continue;
            }
            let step_size = 1. / self.resolution() as f32;
            for t in (0..=self.resolution()).map(|i| i as f32 * step_size) {
                path.push_point(bezier::bezier_cubic(points, t), self.color);
            }
        }
        self.path = path;
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        self.rebuild_mesh_if_needed();
        self.path.draw(frame, position)
    }

    pub fn draw_mode(&self) -> PathDrawingMode {
        self.path.draw_mode()
    }

    pub fn set_draw_mode(&mut self, draw_mode: PathDrawingMode) {
        self.path.set_draw_mode(draw_mode)
    }

    pub fn context(&self) -> &'cx Context {
        self.path.context()
    }
}

#[derive(Debug)]
pub struct BezierPath<'a, 'cx> {
    path: Path<'cx>,
    points: Cow<'a, [Point2<f32>]>,
    color0: Color,
    color1: Color,
    resolution: u32,
    needs_rebuild: bool,
}

impl<'a, 'cx> BezierPath<'a, 'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            path: Path::new(context),
            points: Cow::default(),
            color0: Color::default(),
            color1: Color::default(),
            resolution: 100,
            needs_rebuild: false,
        }
    }

    pub fn points<'b>(&'b self) -> &'b [Point2<f32>]
    where
        'a: 'b,
    {
        &self.points
    }

    pub fn points_mut(&mut self) -> &mut Vec<Point2<f32>> {
        self.needs_rebuild = true;
        let points = mem::take(&mut self.points);
        self.points = points.into_owned().into();
        unsafe { match_into_unchecked!(&mut self.points, Cow::Owned(vec) => vec) }
    }

    pub fn set_resolution(&mut self, resolution: u32) {
        self.needs_rebuild = true;
        self.resolution = resolution;
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn set_color0(&mut self, color0: Color) {
        self.color0 = color0;
    }

    pub fn color0(&self) -> Color {
        self.color0
    }

    pub fn set_color1(&mut self, color1: Color) {
        self.color1 = color1;
    }

    pub fn color1(&self) -> Color {
        self.color1
    }

    fn rebuild_mesh_if_needed(&mut self) {
        if !self.needs_rebuild {
            return;
        }
        self.needs_rebuild = false;
        self.path.clear();
        if self.points().is_empty() {
            return;
        }
        for t in (0..=self.resolution).map(|i| i as f32 / self.resolution as f32) {
            let point = bezier::bezier(self.points(), t);
            let color = self.color0.lerp(self.color1, t);
            self.path.push_point(point, color);
        }
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        self.rebuild_mesh_if_needed();
        self.path.draw(frame, position)
    }

    pub fn draw_mode(&self) -> PathDrawingMode {
        self.path.draw_mode()
    }

    pub fn set_draw_mode(&mut self, draw_mode: PathDrawingMode) {
        self.path.set_draw_mode(draw_mode)
    }

    pub fn context(&self) -> &'cx Context {
        self.path.context()
    }
}

#[derive(Debug)]
pub struct Path<'cx> {
    mesh: Mesh<'cx, ColoredVertex>,
    draw_mode: PathDrawingMode,
}

impl<'cx> Path<'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            mesh: {
                let mut mesh = Mesh::new(context);
                *mesh.primitive_type_mut() = PrimitiveType::LinesList;
                mesh
            },
            draw_mode: PathDrawingMode::Line,
        }
    }

    pub fn clear(&mut self) {
        let (vertices, indices) = self.mesh.vertices_indices_mut();
        vertices.clear();
        indices.clear();
    }

    pub fn context(&self) -> &'cx Context {
        self.mesh.context()
    }

    pub fn draw_mode(&self) -> PathDrawingMode {
        self.draw_mode
    }

    pub fn set_draw_mode(&mut self, draw_mode: PathDrawingMode) {
        self.draw_mode = draw_mode;
        *self.mesh.primitive_type_mut() = match draw_mode {
            PathDrawingMode::Line => glium::index::PrimitiveType::LineStrip,
            PathDrawingMode::Fill => glium::index::PrimitiveType::TriangleFan,
        };
    }

    pub fn push_point(&mut self, position: Point2<f32>, color: Color) {
        let (vertices, indices) = self.mesh.vertices_indices_mut();
        let index = vertices.len() as u32;
        vertices.push(ColoredVertex {
            position: position.into(),
            color: color.into(),
        });
        indices.push(index);
    }

    pub fn n_points(&self) -> usize {
        self.mesh.vertices().len()
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        self.mesh.update_if_needed();
        let (polygon_mode, line_width) = match self.draw_mode() {
            _ if self.n_points() <= 1 => return,
            PathDrawingMode::Line => (glium::PolygonMode::Line, Some(1.0f32)),
            PathDrawingMode::Fill => (glium::PolygonMode::Fill, None),
        };
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
            &glium::DrawParameters {
                polygon_mode,
                line_width,
                backface_culling: glium::BackfaceCullingMode::CullingDisabled,
                ..mesh::default_2d_draw_parameters()
            },
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CircleVertex {
    pub position: [f32; 2],
    pub normalized: [f32; 2],
}

glium::implement_vertex!(CircleVertex, position, normalized);

#[derive(Debug)]
pub struct Circle<'cx> {
    mesh: Mesh<'cx, CircleVertex>,
    fill_color: Color,
    inner_radius: f32,
    outer_radius: f32,
}

impl<'cx> Circle<'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            mesh: Mesh::new(context),
            fill_color: Color::default(),
            inner_radius: f32::default(),
            outer_radius: f32::default(),
        }
    }

    pub fn context(&self) -> &'cx Context {
        self.mesh.context()
    }

    pub fn uniform_fill(&mut self, color: Color) {
        self.fill_color = color;
    }

    pub fn set_outer_radius(&mut self, outer_radius: f32) {
        self.outer_radius = outer_radius;
        self.rebuild_mesh_data();
    }

    pub fn set_inner_radius(&mut self, inner_radius: f32) {
        self.inner_radius = inner_radius;
    }

    pub fn inner_radius(&self) -> f32 {
        self.inner_radius
    }

    pub fn outer_radius(&self) -> f32 {
        self.outer_radius
    }

    pub fn rebuild_mesh_data(&mut self) {
        let rect_size = self.outer_radius * 2.;
        let vertices_data = [
            CircleVertex {
                position: [0., rect_size],
                normalized: [-1., 1.],
            },
            CircleVertex {
                position: [rect_size, 0.],
                normalized: [1., -1.],
            },
            CircleVertex {
                position: [rect_size, rect_size],
                normalized: [1., 1.],
            },
            CircleVertex {
                position: [0., 0.],
                normalized: [-1., -1.],
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

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        if self.fill_color.a.is_zero() {
            return;
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
                color: self.fill_color.into_array(),
                inner_radius_normalized: self.inner_radius / self.outer_radius,
                outer_radius_normalized: 1.0f32,
            },
            &self.context().shader_circle,
            &mesh::default_2d_draw_parameters(),
        );
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RectFillColor {
    Uniform(Color),
    GradientHorizontal(Color, Color),
    GradientVertical(Color, Color),
}

impl From<Color> for RectFillColor {
    fn from(v: Color) -> Self {
        Self::Uniform(v)
    }
}

impl Default for RectFillColor {
    fn default() -> Self {
        Color::default().into()
    }
}

#[derive(Debug)]
pub struct Rect<'cx> {
    mesh: Mesh<'cx, ColoredVertex>,
    fill_color: RectFillColor,
    size: Vector2<f32>,
}

impl<'cx> Rect<'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            mesh: Mesh::new(context),
            fill_color: RectFillColor::default(),
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

    pub fn set_fill_color(&mut self, new_fill_color: RectFillColor) {
        self.fill_color = new_fill_color;
        self.rebuild_mesh_data();
    }

    pub fn uniform_fill(&mut self, color: Color) {
        self.set_fill_color(color.into());
    }

    pub fn gradient_fill_horizontal(&mut self, color_left: Color, color_right: Color) {
        self.set_fill_color(RectFillColor::GradientHorizontal(color_left, color_right));
    }

    pub fn gradient_fill_vertical(&mut self, color_top: Color, color_bottom: Color) {
        self.set_fill_color(RectFillColor::GradientVertical(color_top, color_bottom));
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, position: Point2<f32>) {
        match self.fill_color {
            RectFillColor::Uniform(color) if color.a.is_zero() => return,
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
        let (color_top_left, color_top_right, color_bottom_left, color_bottom_right) = match self
            .fill_color
        {
            RectFillColor::Uniform(color) => (color, color, color, color),
            RectFillColor::GradientHorizontal(color0, color1) => (color0, color1, color0, color1),
            RectFillColor::GradientVertical(color0, color1) => (color0, color0, color1, color1),
        };
        let vertices_data = [
            ColoredVertex {
                position: [0., self.size.y],
                color: color_bottom_left.into_array(),
            },
            ColoredVertex {
                position: [self.size.x, 0.],
                color: color_top_right.into_array(),
            },
            ColoredVertex {
                position: [self.size.x, self.size.y],
                color: color_bottom_right.into_array(),
            },
            ColoredVertex {
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
