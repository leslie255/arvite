use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::mem::take;
use std::path::Path;

use std::ops::Range;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use cgmath::*;

use crate::{
    color::Color,
    context::Context,
    mesh::{self, Mesh, Quad2},
    resource::ResourceLoader,
};

fn uvec2_to_fvec2(uvec: Vector2<u32>) -> Vector2<f32> {
    uvec.map(|x| x as f32)
}

fn normalize_coord_in_texture(texture_size: Vector2<u32>, coord: Vector2<u32>) -> Vector2<f32> {
    uvec2_to_fvec2(coord).div_element_wise(uvec2_to_fvec2(texture_size))
}

/// Description for an atlas font's JSON format.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AtlasFontMetaJson {
    path: String,
    atlas_width: u32,
    atlas_height: u32,
    glyph_width: u32,
    glyph_height: u32,
    present_start: u8,
    present_end: u8,
    glyphs_per_line: u32,
}

pub struct AtlasFont {
    json_path: Option<String>,
    atlas_path: Option<String>,
    present_range: Range<u8>,
    glyphs_per_line: u32,
    glyph_size: Vector2<u32>,
    atlas: DynamicImage,
    pub gl_texture: Option<glium::Texture2d>,
}

impl Debug for AtlasFont {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Font")
            .field("json_path", &self.json_path)
            .field("atlas_path", &self.atlas_path)
            .field("present_range", &self.present_range)
            .field("glyphs_per_line", &self.glyphs_per_line)
            .field("glyph_size", &self.glyph_size)
            .finish_non_exhaustive()
    }
}

impl AtlasFont {
    pub fn load_from_path(
        resource_loader: &ResourceLoader,
        json_subpath: impl AsRef<Path>,
    ) -> Self {
        let json_subpath = json_subpath.as_ref();
        let font_meta = resource_loader.load_json_object::<AtlasFontMetaJson>(json_subpath);
        let atlas_subpath = resource_loader.solve_relative_subpath(json_subpath, &font_meta.path);
        let atlas = resource_loader.load_image(&atlas_subpath);
        Self {
            json_path: Some(json_subpath.as_os_str().to_string_lossy().into_owned()),
            atlas_path: Some(atlas_subpath.as_os_str().to_string_lossy().into_owned()),
            atlas,
            present_range: font_meta.present_start..font_meta.present_end,
            glyphs_per_line: font_meta.glyphs_per_line,
            glyph_size: vec2(font_meta.glyph_width, font_meta.glyph_height),
            gl_texture: None,
        }
    }

    pub fn path(&self) -> Option<&str> {
        self.atlas_path.as_deref()
    }

    pub fn atlas(&self) -> &DynamicImage {
        &self.atlas
    }

    pub fn has_glyph(&self, char: char) -> bool {
        self.present_range.contains(&(char as u8))
    }

    fn position_for_glyph(&self, char: char) -> Vector2<u32> {
        assert!(self.has_glyph(char));
        let ith_glyph = ((char as u8) - self.present_range.start) as u32;
        let glyph_coord = vec2(
            ith_glyph % self.glyphs_per_line,
            ith_glyph / self.glyphs_per_line,
        );
        glyph_coord.mul_element_wise(self.glyph_size)
    }

    pub fn sample(&self, char: char) -> Quad2 {
        let top_left = self.position_for_glyph(char);
        let bottom_right = top_left.add_element_wise(self.glyph_size);
        let atlas_size = vec2(self.atlas.width(), self.atlas.height());
        let top_left = normalize_coord_in_texture(atlas_size, top_left);
        let bottom_right = normalize_coord_in_texture(atlas_size, bottom_right);
        Quad2 {
            left: top_left.x,
            right: bottom_right.x,
            bottom: bottom_right.y,
            top: top_left.y,
        }
    }

    pub fn glyph_aspect_ratio(&self) -> f32 {
        (self.glyph_size.x as f32) / (self.glyph_size.y as f32)
    }

    pub fn glyph_size(&self) -> Vector2<u32> {
        self.glyph_size
    }

    pub fn texture_sampler(&self) -> glium::uniforms::Sampler<glium::Texture2d> {
        mesh::texture_sampler(self.gl_texture.as_ref().unwrap())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacterVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
}

glium::implement_vertex!(CharacterVertex, position, uv);

#[derive(Debug)]
pub struct Line<'a, 'cx> {
    mesh: Mesh<'cx, CharacterVertex>,
    string: Cow<'a, str>,
    shadow: bool,
    fg_color: Color,
    bg_color: Color,
    font_size: f32,
}

impl<'a, 'cx> Line<'a, 'cx> {
    pub fn context(&self) -> &'cx Context {
        self.mesh.context()
    }

    pub fn new(context: &'cx Context) -> Self {
        Self {
            mesh: Mesh::new(context),
            string: "".into(),
            shadow: false,
            fg_color: Color::default(),
            bg_color: Color::default(),
            font_size: 20.,
        }
    }

    pub fn new_with_string(context: &'cx Context, string: Cow<'a, str>) -> Self {
        let mut self_ = Self::new(context);
        self_.set_string(string);
        self_
    }

    pub fn set_string(&mut self, string: Cow<'a, str>) {
        // TODO: this could be more efficient.
        self.clear();
        self.string = String::with_capacity(string.len()).into();
        for char in string.chars() {
            self.push_char(char);
        }
    }

    pub fn push_char(&mut self, char: char) {
        let uv_quad = if self.context().font.has_glyph(char) {
            self.context().font.sample(char)
        } else {
            return;
        };
        // Width is just the aspect ratio because height is 1.
        let glyph_width = self.context().font.glyph_aspect_ratio();
        let quad = Quad2 {
            left: self.string.len() as f32 * glyph_width,
            right: (self.string.len() + 1) as f32 * glyph_width,
            bottom: 1.,
            top: 0.,
        };

        let mut string = Cow::into_owned(take(&mut self.string));
        string.push(char);
        self.string = string.into();

        #[rustfmt::skip]
        let vertices = [
            CharacterVertex { position: [quad.left,  quad.bottom], uv: [uv_quad.left,  uv_quad.bottom] },
            CharacterVertex { position: [quad.right, quad.top   ], uv: [uv_quad.right, uv_quad.top   ] },
            CharacterVertex { position: [quad.right, quad.bottom], uv: [uv_quad.right, uv_quad.bottom] },
            CharacterVertex { position: [quad.left,  quad.top   ], uv: [uv_quad.left,  uv_quad.top   ] },
        ];
        #[rustfmt::skip]
        let indices = [
            0, 1, 2,
            1, 0, 3,
        ];

        self.mesh.append(&vertices, &indices);
    }

    pub fn clear(&mut self) {
        let mut string = take(&mut self.string);
        match &mut string {
            Cow::Borrowed(_) => string = "".into(),
            Cow::Owned(s) => s.clear(),
        }
        self.string = string;
        self.mesh.indices_mut().clear();
        self.mesh.vertices_mut().clear();
    }

    pub fn push_str(&mut self, str: &str) {
        for char in str.chars() {
            self.push_char(char);
        }
    }

    pub fn set_fg_color(&mut self, new_fg_color: Color) {
        self.fg_color = new_fg_color;
    }

    pub fn set_bg_color(&mut self, new_bg_color: Color) {
        self.bg_color = new_bg_color;
    }

    pub fn set_font_size(&mut self, new_font_size: f32) {
        self.font_size = new_font_size;
    }

    pub fn set_shadow(&mut self, new_shadow: bool) {
        self.shadow = new_shadow;
    }

    pub fn fg_color(&mut self) -> Color {
        self.fg_color
    }

    pub fn bg_color(&mut self) -> Color {
        self.bg_color
    }

    pub fn font_size(&mut self) -> f32 {
        self.font_size
    }

    pub fn shadow(&mut self) -> bool {
        self.shadow
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, model: Matrix4<f32>) {
        self.mesh.update_if_needed();
        let model = model * Matrix4::from_nonuniform_scale(self.font_size, self.font_size, 1.);
        let projection = self.context().projection_matrix();
        if self.shadow {
            let model =
                Matrix4::from_translation(vec3(self.font_size * 0.1, self.font_size * 0.1, 0.))
                    * model;
            self.mesh.draw(
                frame,
                glium::uniform! {
                    model: mesh::matrix4_to_array(model),
                    projection: mesh::matrix4_to_array(projection),
                    tex: self.context().font.texture_sampler(),
                    fg_color: [self.fg_color.r * 0.2, self.fg_color.g * 0.2, self.fg_color.b * 0.2, self.fg_color.a],
                    bg_color: Color::default().into_array(),
                },
                &self.context().shader_text,
                &mesh::default_2d_draw_parameters(),
            );
        }
        self.mesh.draw(
            frame,
            glium::uniform! {
                model: mesh::matrix4_to_array(model),
                projection: mesh::matrix4_to_array(projection),
                tex: self.context().font.texture_sampler(),
                fg_color: self.fg_color.into_array(),
                bg_color: self.bg_color.into_array(),
            },
            &self.context().shader_text,
            &mesh::default_2d_draw_parameters(),
        );
    }
}
