use std::{
    borrow::Cow,
    mem::{self},
    time::Instant,
};

use cgmath::*;
use glium::{Texture2d, index::PrimitiveType};

use crate::{
    bezier,
    color::Color,
    context::Context,
    iterator, match_into_unchecked,
    mesh::{self, Mesh},
    truetype::{
        TrueTypeFont,
        glyf::{Curve, SimpleGlyph},
    },
};

fn point_is_nan(point: Point2<f32>) -> bool {
    point.x.is_nan() && point.y.is_nan()
}

/// Vertex used for `Rect` and `Path`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Vertex {
    pub(crate) position: [f32; 2],
}

impl Vertex {
    pub(crate) const fn new(position: [f32; 2]) -> Self {
        Self { position }
    }
}

glium::implement_vertex!(Vertex, position);

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VertexWithUV {
    pub(crate) position: [f32; 2],
    pub(crate) uv: [f32; 2],
}

impl VertexWithUV {
    pub(crate) const fn new(position: [f32; 2], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }
}

glium::implement_vertex!(VertexWithUV, position, uv);

/// Vertex used for `Rect` and `Path`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VertexWithColor {
    pub(crate) position: [f32; 2],
    pub(crate) color: [f32; 4],
}

impl VertexWithColor {
    pub(crate) const fn new(position: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, color }
    }
}

glium::implement_vertex!(VertexWithColor, position, color);

/// Emulated fragment shader that outputs a monochrome color.
fn sd_contours(_position: Vector2<f32>, uv: Vector2<f32>, contours: &[Vec<Vector2<f32>>]) -> f32 {
    #[allow(non_camel_case_types)]
    type vec2 = Vector2<f32>;
    #[allow(non_camel_case_types)]
    type vec3 = Vector3<f32>;
    #[allow(non_camel_case_types)]
    type vec4 = Vector4<f32>;
    #[allow(non_camel_case_types)]
    type bvec2 = Vector2<bool>;
    #[allow(non_camel_case_types)]
    type bvec3 = Vector3<bool>;
    #[allow(non_camel_case_types)]
    type bvec4 = Vector4<bool>;

    #[inline(always)]
    fn clamp(x: f32, min: f32, max: f32) -> f32 {
        x.clamp(min, max)
    }

    #[inline(always)]
    fn length(v: vec2) -> f32 {
        (v.x.powi(2) + v.y.powi(2)).sqrt()
    }

    #[inline(always)]
    fn dot(v0: vec2, v1: vec2) -> f32 {
        v0.dot(v1)
    }

    #[inline(always)]
    fn min(f0: f32, f1: f32) -> f32 {
        f0.min(f1)
    }

    #[inline(always)]
    fn max(f0: f32, f1: f32) -> f32 {
        f0.max(f1)
    }

    #[inline(always)]
    fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
        let x = clamp((x - edge0) / (edge1 - edge0), 0., 1.);
        x.powi(2) * (3. - 2. * x)
    }

    #[inline(always)]
    fn all3(bv: bvec3) -> bool {
        bv.x && bv.y && bv.z
    }

    #[inline(always)]
    fn not3(bv: bvec3) -> bvec3 {
        vec3(!bv.x, !bv.y, !bv.z)
    }

    /// Also returns a flag that's true if it's clockwise.
    fn sd_polygon(v: &[vec2], p: vec2) -> (bool, f32) {
        if v.is_empty() {
            return (false, 0.);
        }
        // The determinant for whether it's clockwise or counter-clockwise.
        let mut dir_det = 0.0;
        let mut d = dot(p - v[0], p - v[0]);
        let mut s = 1.0;
        for i in 0..v.len() {
            let j = match i {
                0 => v.len() - 1,
                i => i - 1,
            };
            let v_i = unsafe { *v.get_unchecked(i) };
            let v_j = unsafe { *v.get_unchecked(j) };
            dir_det += (v_j.x - v_i.x) * (v_j.y + v_i.y);
            let e = v_j - v_i;
            let w = p - v_i;
            let b = w - e * clamp(dot(w, e) / dot(e, e), 0.0, 1.0);
            d = min(d, dot(b, b));
            let c = vec3(p.y >= v_i.y, p.y < v_j.y, e.x * w.y > e.y * w.x);
            if all3(c) || all3(not3(c)) {
                s *= -1.0
            }
        }
        (dir_det.is_sign_positive(), s * d.sqrt())
    }

    let mut sd = 1.0;
    for polygon in contours {
        let (closewise, d) = sd_polygon(polygon, uv);
        if !closewise {
            sd = min(sd, d);
        } else {
            sd = max(sd, -max(sd, d));
        }
    }
    sd
}

#[derive(Debug)]
pub(crate) struct TrueTypeChar<'cx> {
    pub(crate) mesh: Mesh<'cx, VertexWithUV>,
    pub(crate) image: Vec<f32>,
    pub(crate) image_size: Vector2<u32>,
    pub(crate) texture: Texture2d,
    pub(crate) glyph: Option<SimpleGlyph>,
    pub(crate) points: Vec<Vec<Vector2<f32>>>,
}

impl<'cx> TrueTypeChar<'cx> {
    pub(crate) fn new(context: &'cx Context, font: &TrueTypeFont, char: char) -> Self {
        let mut self_ = Self {
            mesh: Mesh::new(context),
            image: Vec::new(),
            image_size: vec2(0, 0),
            texture: Texture2d::empty(&context.display, 1, 1).unwrap(),
            glyph: font.get_glyph(char as u32),
            points: Vec::new(),
        };
        self_.rebuild_mesh_data();
        self_.resample(128, 128);
        self_
    }

    pub(crate) fn context(&self) -> &'cx Context {
        self.mesh.context()
    }

    pub(crate) fn rebuild_mesh_data(&mut self) {
        let Some(ref glyph) = self.glyph else {
            return;
        };
        let glyph_header = glyph.header;
        let glyph_x_min = glyph_header.x_min.to_i16();
        let glyph_y_min = glyph_header.y_min.to_i16();
        let glyph_width = glyph_header
            .x_max
            .to_i16()
            .saturating_sub(glyph_header.x_min.to_i16());
        let glyph_height = glyph_header
            .y_max
            .to_i16()
            .saturating_sub(glyph_header.y_min.to_i16());
        let aspect_ratio = glyph_width as f32 / glyph_height as f32;

        let (vertices, indices) = self.mesh.vertices_indices_mut();
        vertices.clear();
        vertices.extend_from_slice(&[
            VertexWithUV::new([0., 100.], [0., 1.]),
            VertexWithUV::new([100. * aspect_ratio, 0.], [1., 0.]),
            VertexWithUV::new([100. * aspect_ratio, 100.], [1., 1.]),
            VertexWithUV::new([0., 0.], [0., 0.]),
        ]);
        indices.clear();
        indices.extend_from_slice(&[0, 1, 2, 1, 0, 3]);

        let margin = 0.05f32;
        let mut points = Vec::new();
        let convert_point = move |point: Point2<i16>| -> Vector2<f32> {
            vec2(
                (point.x.saturating_sub(glyph_x_min)) as f32 / glyph_width as f32
                    * (1. - 2. * margin)
                    + margin,
                (point.y.saturating_sub(glyph_y_min)) as f32 / glyph_height as f32
                    * (1. - 2. * margin)
                    + margin,
            )
        };
        for contour in glyph.contours() {
            let mut contour_points = Vec::new();
            let mut previous_point = None;
            for &curve in &contour.curves {
                match curve {
                    Curve::Linear(points) => {
                        let [p0, p1] = points.map(convert_point);
                        if previous_point != Some(p0) {
                            contour_points.push(p0);
                        }
                        contour_points.push(p1);
                        previous_point = Some(p1);
                    }
                    Curve::Quadratic(points) => {
                        let [p0, p1, p2] = points.map(convert_point);
                        if previous_point != Some(p0) {
                            contour_points.push(p0);
                        }
                        let resolution = 8;
                        for i in 1..=resolution {
                            let t = i as f32 / resolution as f32;
                            contour_points.push(
                                bezier::bezier_quadratic(
                                    [point2(p0.x, p0.y), point2(p1.x, p1.y), point2(p2.x, p2.y)],
                                    t,
                                )
                                .to_vec(),
                            );
                        }
                        contour_points.push(p2);
                        previous_point = Some(p2);
                    }
                }
            }
            points.push(contour_points);
        }

        self.points = points;
    }

    pub(crate) fn resample(&mut self, width: u32, height: u32) {
        if self.image_size == vec2(width, height) {
            return;
        }

        let before = Instant::now();

        self.image = vec![0.0f32; (width as usize) * (height as usize)];
        self.image_size = vec2(width, height);

        for i_x in 0..width {
            for i_y in (0..height).rev() {
                let position = vec2(i_x as f32, i_y as f32);
                let uv = vec2(position.x / width as f32, position.y / height as f32);
                let fragment = sd_contours(position, uv, &self.points[..]);
                let pixel_index =
                    ((height as usize) - (i_y as usize) - 1) * (width as usize) + (i_x as usize);
                self.image[pixel_index] = fragment;
            }
        }

        let image = glium::texture::RawImage2d {
            data: Cow::Borrowed(self.image.as_ref()),
            width,
            height,
            format: glium::texture::ClientFormat::F32,
        };
        self.texture = Texture2d::with_format(
            &self.context().display,
            image,
            glium::texture::UncompressedFloatFormat::F32,
            glium::texture::MipmapsOption::AutoGeneratedMipmaps,
        )
        .unwrap();

        let after = Instant::now();
        let elapsed = after.duration_since(before);
        println!(
            "[DEBUG] font sampling took {} seconds (width = {width}, height = {height})",
            elapsed.as_secs_f64()
        );
    }

    pub(crate) fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        self.mesh.update_if_needed();
        let model_view = model;
        let projection = self.context().projection_matrix();
        self.mesh.draw(
            frame,
            glium::uniform! {
            model_view: mesh::matrix4_to_array(model_view),
            projection: mesh::matrix4_to_array(projection),
            tex: self.texture
                    .sampled()
                    .magnify_filter(glium::uniforms::MagnifySamplerFilter::Linear)
                    .minify_filter(glium::uniforms::MinifySamplerFilter::Linear)
                    .wrap_function(glium::uniforms::SamplerWrapFunction::Clamp),
            },
            &self.context().shader_test_rect,
            &mesh::default_2d_draw_parameters(),
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathDrawingMode {
    Line,
    Fill,
}

/// A spline made from cubic beziers.
#[derive(Debug)]
pub struct BezierSplinePath<'a, 'cx> {
    path: Path<'cx>,
    segments: Cow<'a, [[Point2<f32>; 3]]>,
    is_closed: bool,
    color: Color,
    resolution: u32,
    needs_rebuild: bool,
}

impl<'a, 'cx> BezierSplinePath<'a, 'cx> {
    pub fn new(context: &'cx Context) -> Self {
        Self {
            path: Path::new(context),
            segments: Cow::default(),
            is_closed: true,
            color: Color::default(),
            resolution: 48,
            needs_rebuild: false,
        }
    }

    pub(crate) fn path(&self) -> &Path<'cx> {
        &self.path
    }

    pub fn segments<'b>(&'b self) -> &'b [[Point2<f32>; 3]]
    where
        'a: 'b,
    {
        &self.segments
    }

    pub fn segments_mut(&mut self) -> &mut Vec<[Point2<f32>; 3]> {
        self.needs_rebuild = true;
        let points = mem::take(&mut self.segments);
        self.segments = points.into_owned().into();
        unsafe { match_into_unchecked!(&mut self.segments, Cow::Owned(vec) => vec) }
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

    pub fn set_is_closed(&mut self, is_closed: bool) {
        self.needs_rebuild = true;
        self.is_closed = is_closed;
    }

    pub fn is_closed(&self) -> bool {
        self.is_closed
    }

    /// Similar to SVG path's `L` command.
    /// If no current points exists, adds a new point at (0, 0,).
    pub fn append_linear(&mut self, point: Point2<f32>) {
        let segments = self.segments_mut();
        match segments.last_mut() {
            Some(last) => last[2] = last[1],
            None => segments.push([point2(0., 0.), point2(0., 0.), point2(0., 0.)]),
        }
        segments.push([point, point, point]);
    }

    /// Similar to SVG path's `Q` command.
    /// If no current points exists, adds a new point at (0, 0,).
    pub fn append_quadratic(&mut self, points: [Point2<f32>; 2]) {
        let segments = self.segments_mut();
        match segments.last_mut() {
            Some(last) => last[2] = points[0],
            None => segments.push([point2(0., 0.), point2(0., 0.), point2(0., 0.)]),
        }
        segments.push([points[0], points[1], points[1]]);
    }

    /// Similar to SVG path's `C` command.
    /// If no current points exists, adds a new point at (0, 0,).
    pub fn append_cubic(&mut self, points: [Point2<f32>; 3]) {
        let segments = self.segments_mut();
        match segments.last_mut() {
            Some(last) => last[2] = points[0],
            None => segments.push([point2(0., 0.), point2(0., 0.), point2(0., 0.)]),
        }
        segments.push([points[1], points[2], points[2]]);
    }

    /// Iterator through every cubic beziers that makes up this spline.
    pub fn cubics(&self) -> impl Iterator<Item = [Point2<f32>; 4]> {
        iterator! {
            for &[this, next] in self.segments().array_windows::<2>() {
                yield [this[1], this[2], next[0], next[1]];
            }
            if self.segments().len() >= 3 && self.is_closed {
                let first = self.segments().first().unwrap();
                let last = self.segments().last().unwrap();
                yield [last[1], last[2], first[0], first[1]]
            }
        }
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
        let step_size = 1. / self.resolution() as f32;
        for points in self.cubics() {
            if points[1] == points[0] && points[2] == points[3] {
                // Linear.
                path.push_point(points[0], self.color);
                path.push_point(points[3], self.color);
            } else if points[0] == points[1] || points[2] == points[3] || points[1] == points[2] {
                // Quadratic.
                let points = match points {
                    // Compiler would probably optimize this.
                    _ if points[0] == points[1] => [points[0], points[2], points[3]],
                    _ if points[2] == points[3] => [points[0], points[1], points[2]],
                    _ if points[1] == points[2] => [points[0], points[1], points[3]],
                    _ => unreachable!(),
                };
                for t in (0..=self.resolution()).map(|i| i as f32 * step_size) {
                    path.push_point(bezier::bezier_quadratic(points, t), self.color);
                }
            } else {
                // Cubic.
                for t in (0..=self.resolution()).map(|i| i as f32 * step_size) {
                    path.push_point(bezier::bezier_cubic(points, t), self.color);
                }
            }
        }
        self.path = path;
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        self.rebuild_mesh_if_needed();
        self.path.draw(frame, model)
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

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        self.rebuild_mesh_if_needed();
        self.path.draw(frame, model)
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
    mesh: Mesh<'cx, VertexWithColor>,
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
        vertices.push(VertexWithColor {
            position: position.into(),
            color: color.into(),
        });
        indices.push(index);
    }

    pub fn n_points(&self) -> usize {
        self.mesh.vertices().len()
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        self.mesh.update_if_needed();
        let (polygon_mode, line_width) = match self.draw_mode() {
            _ if self.n_points() <= 1 => return,
            PathDrawingMode::Line => (glium::PolygonMode::Line, Some(1.0f32)),
            PathDrawingMode::Fill => (glium::PolygonMode::Fill, None),
        };
        let projection = self.context().projection_matrix();
        let model_view = model;
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

#[derive(Debug)]
pub struct Circle<'cx> {
    mesh: Mesh<'cx, VertexWithUV>,
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

    pub(crate) fn rebuild_mesh_data(&mut self) {
        let rect_size = self.outer_radius * 2.;
        let vertices_data = [
            VertexWithUV::new([0., rect_size], [-1., 1.]),
            VertexWithUV::new([rect_size, 0.], [1., -1.]),
            VertexWithUV::new([rect_size, rect_size], [1., 1.]),
            VertexWithUV::new([0., 0.], [-1., -1.]),
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

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        if self.fill_color.a.is_zero() {
            return;
        }
        self.mesh.update_if_needed();
        let model_view = model;
        let projection = self.context().projection_matrix();
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
    mesh: Mesh<'cx, VertexWithColor>,
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

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        match self.fill_color {
            RectFillColor::Uniform(color) if color.a.is_zero() => return,
            _ => (),
        }
        self.mesh.update_if_needed();
        let model_view = model;
        let projection = self.context().projection_matrix();
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
            VertexWithColor::new([0., self.size.y], color_bottom_left.into_array()),
            VertexWithColor::new([self.size.x, 0.], color_top_right.into_array()),
            VertexWithColor::new([self.size.x, self.size.y], color_bottom_right.into_array()),
            VertexWithColor::new([0., 0.], color_top_left.into_array()),
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
