use cgmath::*;

use crate::{resource::ResourceLoader, text::AtlasFont};

#[derive(Debug)]
pub struct Context {
    pub(crate) loader: ResourceLoader,
    pub(crate) shader_text: glium::Program,
    pub(crate) shader_rect: glium::Program,
    pub(crate) shader_circle: glium::Program,
    pub(crate) font: AtlasFont,
    pub(crate) display: glium::Display<glium::glutin::surface::WindowSurface>,
}

impl Context {
    pub fn new(display: glium::Display<glium::glutin::surface::WindowSurface>) -> Self {
        let loader = ResourceLoader::with_default_res_directory().unwrap();
        Self {
            shader_text: Self::load_shader(&display, &loader, "shader/text"),
            shader_rect: Self::load_shader(&display, &loader, "shader/rect"),
            shader_circle: Self::load_shader(&display, &loader, "shader/circle"),
            font: Self::load_font(&display, &loader, "font/big_blue_terminal.json"),
            loader,
            display,
        }
    }

    fn load_texture(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> glium::Texture2d {
        let image = resource_loader.load_image(name);
        let image = glium::texture::RawImage2d::from_raw_rgba(
            image.to_rgba8().into_raw(),
            (image.width(), image.height()),
        );
        glium::Texture2d::new(display, image).unwrap()
    }

    /// `name` is in the format of `"shader/name"`, which would load `"res/shader/name.vs"` and `"res/shader/name.fs"`.
    fn load_shader(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> glium::Program {
        let vs_source = resource_loader.read_to_string(format!("{name}.vs"));
        let fs_source = resource_loader.read_to_string(format!("{name}.fs"));
        glium::Program::from_source(display, &vs_source, &fs_source, None).unwrap()
    }

    fn load_font(
        display: &impl glium::backend::Facade,
        resource_loader: &ResourceLoader,
        name: &str,
    ) -> AtlasFont {
        let mut font = AtlasFont::load_from_path(resource_loader, name);
        let image = glium::texture::RawImage2d::from_raw_rgba(
            font.atlas().to_rgba8().into_raw(),
            (font.atlas().width(), font.atlas().height()),
        );
        font.gl_texture = Some(glium::Texture2d::new(display, image).unwrap());
        font
    }

    /// Notify it of a resize event.
    pub fn resize(&self, new_size: Vector2<u32>) {
        self.display.resize(new_size.into());
    }

    pub fn display_size(&self) -> Vector2<f32> {
        Vector2::from(self.display.get_framebuffer_dimensions()).map(|x| x as f32)
    }
}


