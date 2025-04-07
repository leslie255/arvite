use std::{borrow::Cow, ops::RangeInclusive, sync::Arc};

use crate::bind_group::BindGroupBuilder;
use crate::buffer::{IndexBuffer, UniformBuffer, Vertex2d, Vertex2dUV, VertexBuffer};
use crate::context::{Context, Surface};
use crate::include_wgsl;
use crate::pipeline::PipelineBuilder;
use crate::texture::{SamplerBuilder, Texture2d, TextureView2d};

use cgmath::*;

const SQUARE_VERTICES: [Vertex2d; 4] = [
    Vertex2d::new([0.0, 0.0]),
    Vertex2d::new([1.0, 0.0]),
    Vertex2d::new([1.0, 1.0]),
    Vertex2d::new([0.0, 1.0]),
];

/// Square vertices with texture-space UV.
const SQUARE_VERTICES_UV: [Vertex2dUV; 4] = [
    Vertex2dUV::new([0.0, 0.0], [0.0, 1.0]),
    Vertex2dUV::new([1.0, 0.0], [1.0, 1.0]),
    Vertex2dUV::new([1.0, 1.0], [1.0, 0.0]),
    Vertex2dUV::new([0.0, 1.0], [0.0, 0.0]),
];

/// Square vertices with NDC-like UV mapping. Where center is (0,0).
const CIRCLE_VERTICES_NDC: [Vertex2dUV; 4] = [
    Vertex2dUV::new([-1.0, -1.0], [-1.0, -1.0]),
    Vertex2dUV::new([1.0, -1.0], [1.0, -1.0]),
    Vertex2dUV::new([1.0, 1.0], [1.0, 1.0]),
    Vertex2dUV::new([-1.0, 1.0], [-1.0, 1.0]),
];

const SQUARE_INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

#[derive(Debug)]
pub struct ClassicFont {
    #[expect(dead_code)]
    pub(crate) atlas: &'static [u8],
    pub(crate) atlas_size: Vector2<u32>,
    pub(crate) glyph_size: Vector2<u32>,
    pub(crate) present_range: RangeInclusive<u8>,
    pub(crate) glyphs_per_line: u32,
    pub(crate) texture: Arc<Texture2d>,
}

impl ClassicFont {
    pub fn big_blue_terminal(context: &Context) -> Self {
        let atlas = include_bytes!("./big_blue_terminal_atlas.bin");
        let atlas_width = 128;
        let atlas_height = 128;
        Self {
            atlas,
            atlas_size: vec2(atlas_width, atlas_height),
            glyph_size: vec2(8, 12),
            present_range: 32..=126,
            glyphs_per_line: 16,
            texture: {
                let texture = Texture2d::new(
                    context,
                    vec2(atlas_width, atlas_height),
                    wgpu::TextureFormat::R8Unorm,
                );
                texture.write(context, atlas);
                Arc::new(texture)
            },
        }
    }

    pub fn texture(&self) -> Arc<Texture2d> {
        Arc::clone(&self.texture)
    }

    pub fn has_glyph(&self, char: char) -> bool {
        match u8::try_from(char as u32) {
            Ok(u) => self.present_range.contains(&u),
            Err(_) => false,
        }
    }

    pub fn position_for_glyph(&self, char: char) -> Option<Vector2<u32>> {
        if !self.has_glyph(char) {
            return None;
        }
        let range_start = *self.present_range.start();
        let ith_glyph = ((char as u8).checked_sub(range_start)?) as u32;
        let glyph_coord = vec2(
            ith_glyph.checked_rem(self.glyphs_per_line)?,
            ith_glyph.checked_div(self.glyphs_per_line)?,
        );
        Some(glyph_coord.mul_element_wise(self.glyph_size))
    }

    /// Returns the top left and bottom right corners, with the bottom right coordinate being one-past.
    pub fn quad_for_glyph(&self, char: char) -> Option<(Vector2<u32>, Vector2<u32>)> {
        let top_left = self.position_for_glyph(char)?;
        Some((top_left, top_left + self.glyph_size))
    }

    /// Returns the top left and bottom right corners, in texture-space UV coordinates.
    pub fn quad_for_glyph_uv(&self, char: char) -> Option<(Vector2<f32>, Vector2<f32>)> {
        let (top_left, bottom_right) = self.quad_for_glyph(char)?;

        let top_left_f32 = top_left.map(|x| x as f32);
        let bottom_right_f32 = bottom_right.map(|x| x as f32);
        let atlas_size_f32 = self.atlas_size.map(|x| x as f32);

        let top_left_uv = top_left_f32.div_element_wise(atlas_size_f32);
        let bottom_right_uv = bottom_right_f32.div_element_wise(atlas_size_f32);

        Some((top_left_uv, bottom_right_uv))
    }

    pub fn glyph_aspect_ratio(&self) -> f32 {
        (self.glyph_size.x as f32) / (self.glyph_size.y as f32)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ClassicTextMesh {
    pub(crate) render_pipeline: Arc<wgpu::RenderPipeline>,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) vertex_buffer: VertexBuffer<Vertex2dUV>,
    pub(crate) index_buffer: IndexBuffer<u16>,
    pub(crate) n_indices: u32,
    pub(crate) uniform0_model_view: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform1_projection: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform2_fg_color: UniformBuffer<[f32; 4]>,
}

impl ClassicTextMesh {
    fn vertices_for_char(
        font: &ClassicFont,
        char: char,
        position: Vector2<f32>,
    ) -> [Vertex2dUV; 4] {
        let (uv_min, uv_max) = font.quad_for_glyph_uv(char).unwrap();
        [
            Vertex2dUV::new((position + vec2(0.0, -1.0)).into(), [uv_min.x, uv_max.y]),
            Vertex2dUV::new((position + vec2(1.0, -1.0)).into(), [uv_max.x, uv_max.y]),
            Vertex2dUV::new((position + vec2(1.0, 0.0)).into(), [uv_max.x, uv_min.y]),
            Vertex2dUV::new((position + vec2(0.0, 0.0)).into(), [uv_min.x, uv_min.y]),
        ]
    }

    fn new(context: &Context, surface: &Surface, font: &ClassicFont, text: &str) -> Self {
        let mut vertices = Vec::<Vertex2dUV>::new();
        let mut indices = Vec::<u16>::new();
        let mut i_x = 0i32;
        let mut i_y = 0i32;
        for char in text.chars() {
            match char {
                '\n' => {
                    i_x = 0;
                    i_y -= 1;
                }
                char => {
                    let char_vertices =
                        Self::vertices_for_char(font, char, vec2(i_x as f32, i_y as f32));
                    let char_indices = SQUARE_INDICES.map(|x| x + (vertices.len() as u16));
                    vertices.extend_from_slice(&char_vertices);
                    indices.extend_from_slice(&char_indices);
                    i_x += 1;
                }
            }
        }
        let vertex_buffer = VertexBuffer::new_initialized(context, &vertices);
        let index_buffer = IndexBuffer::new_initialized(context, &indices);
        let n_indices = indices.len() as u32;

        let uniform0_model_view = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform1_projection = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform2_fg_color = UniformBuffer::<[f32; 4]>::new_zeroed(context);

        let texture_view = font.texture.create_view();
        let sampler = SamplerBuilder::new(context)
            .mag_filter(wgpu::FilterMode::Nearest)
            .min_filter(wgpu::FilterMode::Linear)
            .clamp(wgpu::AddressMode::ClampToEdge)
            .build();

        let pipeline = || {
            let shader = include_wgsl!(context, "./shaders/classic_text.wgsl");
            let mut builder = PipelineBuilder::new(context, &shader);
            builder.texture_format(surface.format);
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_fg_color);
            builder.bind_resource(3, &texture_view);
            builder.bind_resource(4, &sampler);
            builder.add_vertex_buffer(&vertex_buffer);
            builder.build()
        };

        let render_pipeline = surface
            .pipeline_reuse_pool()
            .lookup_or_insert_with("classic_text", pipeline);

        let bind_group = {
            let layout = render_pipeline.get_bind_group_layout(0);
            let mut builder = BindGroupBuilder::new(context, &layout);
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_fg_color);
            builder.bind_resource(3, &texture_view);
            builder.bind_resource(4, &sampler);
            builder.build()
        };

        Self {
            render_pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            n_indices,
            uniform0_model_view,
            uniform1_projection,
            uniform2_fg_color,
        }
    }

    fn draw(
        &self,
        context: &Context,
        surface: &Surface,
        render_pass: &mut wgpu::RenderPass,
        model: Matrix4<f32>,
        fg_color: Vector4<f32>,
    ) {
        render_pass.push_debug_group("ClassicText::draw");

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        self.index_buffer.set(render_pass, ..);
        self.vertex_buffer.set(render_pass, 0, ..);

        self.uniform0_model_view.write(context, model.into());
        self.uniform1_projection
            .write(context, surface.projection_matrix().into());
        self.uniform2_fg_color.write(context, fg_color.into());

        render_pass.draw_indexed(0..self.n_indices, 0, 0..1);

        render_pass.pop_debug_group();
    }
}

/// An ASCII-only, monospaced text rendered with a texture atlas.
#[derive(Debug, Clone)]
pub struct ClassicText<'cx> {
    pub(crate) context: &'cx Context,

    /// `None` when string is empty.
    /// Lazy updated on `draw`.
    pub(crate) mesh: Option<ClassicTextMesh>,

    pub(crate) font: Arc<ClassicFont>,

    pub(crate) text_height: f32,
    pub(crate) fg_color: Vector4<f32>,
    pub(crate) string: String,
    pub(crate) string_changed: bool,
}

impl<'cx> ClassicText<'cx> {
    pub fn new(context: &'cx Context, surface: &Surface) -> Self {
        _ = surface;
        let font = context.big_blue_terminal_font();

        Self {
            context,
            font,
            mesh: None,
            text_height: 24.0,
            fg_color: vec4(1.0, 1.0, 1.0, 1.0),
            string: String::new(),
            string_changed: false,
        }
    }

    /// Transforms from local space to pixel space.
    fn model_matrix(&self) -> Matrix4<f32> {
        let height = self.text_height;
        let width = self.text_height * self.font.glyph_aspect_ratio();
        Matrix4::from_nonuniform_scale(width, height, 1.0)
    }

    pub fn draw(
        &mut self,
        surface: &Surface,
        render_pass: &mut wgpu::RenderPass,
        model: Matrix4<f32>,
    ) {
        let model = model * self.model_matrix();
        if self.string_changed {
            self.string_changed = false;
            self.mesh = if self.string.is_empty() {
                None
            } else {
                Some(ClassicTextMesh::new(
                    self.context,
                    surface,
                    &self.font,
                    &self.string,
                ))
            };
        }
        if let Some(ref mut mesh) = self.mesh {
            mesh.draw(self.context, surface, render_pass, model, self.fg_color);
        }
    }

    pub fn string(&self) -> &str {
        &self.string
    }

    pub fn string_mut(&mut self) -> &mut String {
        self.string_changed = true;
        &mut self.string
    }

    pub fn fg_color(&self) -> Vector4<f32> {
        self.fg_color
    }

    pub fn fg_color_mut(&mut self) -> &mut Vector4<f32> {
        &mut self.fg_color
    }

    pub fn set_fg_color(&mut self, fg_color: Vector4<f32>) {
        self.fg_color = fg_color;
    }

    pub fn text_height(&self) -> f32 {
        self.text_height
    }

    pub fn text_height_mut(&mut self) -> &mut f32 {
        &mut self.text_height
    }

    pub fn set_text_height(&mut self, text_height: f32) {
        self.text_height = text_height;
    }
}

#[derive(Debug, Clone)]
pub struct Circle<'cx> {
    pub(crate) context: &'cx Context,

    pub(crate) render_pipeline: Arc<wgpu::RenderPipeline>,
    pub(crate) bind_group: wgpu::BindGroup,

    pub(crate) vertex_buffer: VertexBuffer<Vertex2dUV>,
    pub(crate) index_buffer: IndexBuffer<u16>,

    pub(crate) uniform0_model_view: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform1_projection: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform2_fill_color: UniformBuffer<[f32; 4]>,
    pub(crate) uniform3_inner_radius: UniformBuffer<f32>,

    pub(crate) outer_radius: f32,
    pub(crate) inner_radius: f32,
    pub(crate) fill_color: Vector4<f32>,
}

impl<'cx> Circle<'cx> {
    pub fn new(context: &'cx Context, surface: &Surface) -> Self {
        let vertex_buffer = VertexBuffer::new_initialized(context, &CIRCLE_VERTICES_NDC);
        let index_buffer = IndexBuffer::new_initialized(context, &SQUARE_INDICES);

        let uniform0_model_view = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform1_projection = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform2_fill_color = UniformBuffer::<[f32; 4]>::new_zeroed(context);
        let uniform3_inner_radius = UniformBuffer::<f32>::new_zeroed(context);

        let pipeline = || {
            let shader = include_wgsl!(context, "./shaders/circle.wgsl");
            let mut builder = PipelineBuilder::new(context, &shader);
            builder.texture_format(surface.format);
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_fill_color);
            builder.bind_resource(3, &uniform3_inner_radius);
            builder.add_vertex_buffer(&vertex_buffer);
            builder.build()
        };

        let render_pipeline = surface
            .pipeline_reuse_pool()
            .lookup_or_insert_with("circle", pipeline);

        let bind_group = {
            let layout = render_pipeline.get_bind_group_layout(0);
            let mut builder = BindGroupBuilder::new(context, &layout);
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_fill_color);
            builder.bind_resource(3, &uniform3_inner_radius);
            builder.build()
        };

        Self {
            context,
            render_pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            uniform0_model_view,
            uniform1_projection,
            uniform2_fill_color,
            uniform3_inner_radius,
            outer_radius: 100.0,
            inner_radius: 50.0,
            fill_color: vec4(1.0, 1.0, 1.0, 1.0),
        }
    }

    /// Transforms from local space to pixel space.
    fn model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_scale(self.outer_radius)
    }

    pub fn draw(
        &mut self,
        surface: &Surface,
        render_pass: &mut wgpu::RenderPass,
        model: Matrix4<f32>,
    ) {
        render_pass.push_debug_group("Cricle::draw");

        // Pipeline.
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        // Vertex/index buffer.
        self.index_buffer.set(render_pass, ..);
        self.vertex_buffer.set(render_pass, 0, ..);

        // Uniforms.
        self.uniform0_model_view
            .write(self.context, (model * self.model_matrix()).into());
        self.uniform1_projection
            .write(self.context, surface.projection_matrix().into());
        self.uniform2_fill_color
            .write(self.context, self.fill_color.into());
        self.uniform3_inner_radius
            .write(self.context, self.inner_radius / self.outer_radius);

        // Draw.
        render_pass.draw_indexed(0..(SQUARE_INDICES.len() as u32), 0, 0..1);

        render_pass.pop_debug_group();
    }

    pub fn outer_radius(&self) -> f32 {
        self.outer_radius
    }

    pub fn outer_radius_mut(&mut self) -> &mut f32 {
        &mut self.outer_radius
    }

    pub fn with_outer_radius(mut self, outer_radius: f32) -> Self {
        *self.outer_radius_mut() = outer_radius;
        self
    }

    pub fn inner_radius(&self) -> f32 {
        self.inner_radius
    }

    pub fn inner_radius_mut(&mut self) -> &mut f32 {
        &mut self.inner_radius
    }

    pub fn with_inner_radius(mut self, inner_radius: f32) -> Self {
        *self.inner_radius_mut() = inner_radius;
        self
    }

    pub fn fill_color(&self) -> Vector4<f32> {
        self.fill_color
    }

    pub fn fill_color_mut(&mut self) -> &mut Vector4<f32> {
        &mut self.fill_color
    }

    pub fn with_fill_color(mut self, fill_color: Vector4<f32>) -> Self {
        *self.fill_color_mut() = fill_color;
        self
    }
}

#[derive(Debug, Clone)]
pub struct TexturedRectangle<'cx> {
    /// The debug label.
    pub label: Option<Cow<'static, str>>,

    pub(crate) context: &'cx Context,

    pub(crate) render_pipeline: Arc<wgpu::RenderPipeline>,
    pub(crate) bind_group: wgpu::BindGroup,

    pub(crate) vertex_buffer: VertexBuffer<Vertex2dUV>,
    pub(crate) index_buffer: IndexBuffer<u16>,

    pub(crate) uniform0_model_view: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform1_projection: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform2_gamma: UniformBuffer<f32>,

    pub(crate) size: Vector2<f32>,
    pub(crate) gamma: f32,
}

impl<'cx> TexturedRectangle<'cx> {
    pub fn new(context: &'cx Context, surface: &Surface, texture_view: TextureView2d) -> Self {
        let vertex_buffer = VertexBuffer::new_initialized(context, &SQUARE_VERTICES_UV);
        let index_buffer = IndexBuffer::new_initialized(context, &SQUARE_INDICES);

        let uniform0_model_view = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform1_projection = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform2_gamma = UniformBuffer::<f32>::new_zeroed(context);

        let sampler = SamplerBuilder::new(context)
            .mag_filter(wgpu::FilterMode::Linear)
            .min_filter(wgpu::FilterMode::Linear)
            .clamp(wgpu::AddressMode::ClampToEdge)
            .build();

        let pipeline = || {
            let shader = include_wgsl!(context, "./shaders/rectangle_textured.wgsl");
            let mut builder = PipelineBuilder::new(context, &shader);
            builder.texture_format(surface.format());
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_gamma);
            builder.bind_resource(3, &texture_view);
            builder.bind_resource(4, &sampler);
            builder.add_vertex_buffer(&vertex_buffer);
            builder.build()
        };

        let render_pipeline = surface
            .pipeline_reuse_pool()
            .lookup_or_insert_with("textured_rectangle", pipeline);

        let bind_group = {
            let layout = render_pipeline.get_bind_group_layout(0);
            let mut builder = BindGroupBuilder::new(context, &layout);
            builder.bind_resource(0, &uniform0_model_view);
            builder.bind_resource(1, &uniform1_projection);
            builder.bind_resource(2, &uniform2_gamma);
            builder.bind_resource(3, &texture_view);
            builder.bind_resource(4, &sampler);
            builder.build()
        };

        Self {
            label: None,
            context,
            render_pipeline,
            bind_group,
            vertex_buffer,
            index_buffer,
            uniform0_model_view,
            uniform1_projection,
            uniform2_gamma,
            size: vec2(100.0, 100.0),
            gamma: 1.0,
        }
    }

    fn debug_group_name(&self) -> Cow<str> {
        match self.label {
            Some(ref label) => format!("Rectangle::draw (instance label: {label})").into(),
            None => "Rectangle::draw".into(),
        }
    }

    /// Transforms from local space to pixel space.
    fn model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_nonuniform_scale(self.size.x, self.size.y, 1.0)
    }

    pub fn draw(
        &mut self,
        surface: &Surface,
        render_pass: &mut wgpu::RenderPass,
        model: Matrix4<f32>,
    ) {
        render_pass.push_debug_group(&self.debug_group_name());

        // Pipeline.
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        // Vertex/index buffer.
        self.index_buffer.set(render_pass, ..);
        self.vertex_buffer.set(render_pass, 0, ..);

        // Uniforms.
        self.uniform0_model_view
            .write(self.context, (model * self.model_matrix()).into());
        self.uniform1_projection
            .write(self.context, surface.projection_matrix().into());
        self.uniform2_gamma.write(self.context, self.gamma);

        // Draw.
        render_pass.draw_indexed(0..(SQUARE_INDICES.len() as u32), 0, 0..1);

        render_pass.pop_debug_group();
    }

    pub fn size(&self) -> Vector2<f32> {
        self.size
    }

    pub fn size_mut(&mut self) -> &mut Vector2<f32> {
        &mut self.size
    }

    pub fn with_size(mut self, size: Vector2<f32>) -> Self {
        *self.size_mut() = size;
        self
    }

    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    pub fn gamma_mut(&mut self) -> &mut f32 {
        &mut self.gamma
    }

    pub fn with_gamma(mut self, size: f32) -> Self {
        *self.gamma_mut() = size;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Rectangle<'cx> {
    /// The debug label.
    pub label: Option<Cow<'static, str>>,

    pub(crate) context: &'cx Context,

    pub(crate) render_pipeline: Arc<wgpu::RenderPipeline>,
    pub(crate) bind_group: wgpu::BindGroup,

    pub(crate) vertex_buffer: VertexBuffer<Vertex2d>,
    pub(crate) index_buffer: IndexBuffer<u16>,

    pub(crate) uniform0_fill_color: UniformBuffer<[f32; 4]>,
    pub(crate) uniform1_model_view: UniformBuffer<[[f32; 4]; 4]>,
    pub(crate) uniform2_projection: UniformBuffer<[[f32; 4]; 4]>,

    pub(crate) fill_color: Vector4<f32>,
    pub(crate) size: Vector2<f32>,
}

impl<'cx> Rectangle<'cx> {
    pub fn new(context: &'cx Context, surface: &Surface) -> Self {
        // Vertex and index buffer.
        let vertex_buffer = VertexBuffer::new_initialized(context, &SQUARE_VERTICES);
        let index_buffer = IndexBuffer::new_initialized(context, &SQUARE_INDICES);

        // Uniforms.
        let uniform0_fill_color = UniformBuffer::<[f32; 4]>::new_zeroed(context);
        let uniform1_model_view = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);
        let uniform2_projection = UniformBuffer::<[[f32; 4]; 4]>::new_zeroed(context);

        let pipeline = || {
            let shader = include_wgsl!(context, "./shaders/rectangle.wgsl");
            let mut builder = PipelineBuilder::new(context, &shader);
            builder.texture_format(surface.format);
            builder.bind_resource(0, &uniform0_fill_color);
            builder.bind_resource(1, &uniform1_model_view);
            builder.bind_resource(2, &uniform2_projection);
            builder.add_vertex_buffer(&vertex_buffer);
            builder.build()
        };

        let render_pipeline = surface
            .pipeline_reuse_pool()
            .lookup_or_insert_with("rectangle", pipeline);

        let bind_group = {
            let layout = render_pipeline.get_bind_group_layout(0);
            let mut builder = BindGroupBuilder::new(context, &layout);
            builder.bind_resource(0, &uniform0_fill_color);
            builder.bind_resource(1, &uniform1_model_view);
            builder.bind_resource(2, &uniform2_projection);
            builder.build()
        };

        Self {
            context,
            label: None,
            render_pipeline,
            bind_group,
            uniform0_fill_color,
            uniform1_model_view,
            uniform2_projection,
            vertex_buffer,
            index_buffer,
            fill_color: vec4(1.0, 1.0, 1.0, 1.0),
            size: vec2(1., 1.),
        }
    }

    fn debug_group_name(&self) -> Cow<str> {
        match self.label {
            Some(ref label) => format!("Rectangle::draw (instance label: {label})").into(),
            None => "Rectangle::draw".into(),
        }
    }

    /// Transforms from local space to pixel space.
    fn model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_nonuniform_scale(self.size.x, self.size.y, 1.0)
    }

    pub fn draw(
        &mut self,
        surface: &Surface,
        render_pass: &mut wgpu::RenderPass,
        model: Matrix4<f32>,
    ) {
        render_pass.push_debug_group(&self.debug_group_name());

        // Pipeline.
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);

        // Vertex/index buffer.
        self.index_buffer.set(render_pass, ..);
        self.vertex_buffer.set(render_pass, 0, ..);

        // Uniforms.
        self.uniform0_fill_color
            .write(self.context, self.fill_color.into());
        self.uniform1_model_view
            .write(self.context, (model * self.model_matrix()).into());
        self.uniform2_projection
            .write(self.context, surface.projection_matrix().into());

        // Draw.
        render_pass.draw_indexed(0..(SQUARE_INDICES.len() as u32), 0, 0..1);

        render_pass.pop_debug_group();
    }

    pub fn fill_color(&self) -> Vector4<f32> {
        self.fill_color
    }

    pub fn fill_color_mut(&mut self) -> &mut Vector4<f32> {
        &mut self.fill_color
    }

    pub fn with_fill_color(mut self, fill_color: Vector4<f32>) -> Self {
        *self.fill_color_mut() = fill_color;
        self
    }

    pub fn size(&self) -> Vector2<f32> {
        self.size
    }

    pub fn size_mut(&mut self) -> &mut Vector2<f32> {
        &mut self.size
    }

    pub fn with_size(mut self, size: Vector2<f32>) -> Self {
        *self.size_mut() = size;
        self
    }
}
