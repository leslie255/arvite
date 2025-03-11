use std::time::{Duration, Instant, SystemTime};

use cgmath::*;
use glium::{
    Surface,
    winit::{
        self,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{
    bezier,
    color::Color,
    input::InputHelper,
    rect::Rect,
    resource::ResourceLoader,
    text::{AtlasFont, Line},
};

pub trait GliumBackend: glium::backend::Facade + Surface {}

impl<T: glium::backend::Facade + Surface + ?Sized> GliumBackend for T {}

#[derive(Debug)]
pub struct Context {
    pub(crate) loader: ResourceLoader,
    pub(crate) shader_text: glium::Program,
    pub(crate) shader_rect: glium::Program,
    pub(crate) font: AtlasFont,
    pub(crate) display: glium::Display<glium::glutin::surface::WindowSurface>,
}

impl Context {
    pub fn load(display: glium::Display<glium::glutin::surface::WindowSurface>) -> Self {
        let loader = ResourceLoader::with_default_res_directory().unwrap();
        Self {
            shader_text: Self::load_shader(&display, &loader, "shader/text"),
            shader_rect: Self::load_shader(&display, &loader, "shader/rect"),
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

    fn resize(&self, new_size: Vector2<u32>) {
        self.display.resize(new_size.into());
    }

    pub fn display_size(&self) -> Vector2<f32> {
        Vector2::from(self.display.get_framebuffer_dimensions()).map(|x| x as f32)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FpsCounter {
    /// Sets the update interval.
    pub update_interval: Duration,
    last_update: Instant,
    counter: u32,
    fps: f64,
}

impl FpsCounter {
    pub fn new() -> Self {
        Self {
            update_interval: Duration::from_secs_f64(0.5),
            last_update: Instant::now(),
            counter: 0,
            fps: f64::NAN,
        }
    }

    /// Should be called after every frame is drawn.
    /// Returns `Some(fps)` if FPS is updated.
    pub fn frame(&mut self) -> Option<f64> {
        self.counter += 1;
        let now = Instant::now();
        let since_last_update = now.duration_since(self.last_update);
        if since_last_update > self.update_interval {
            self.fps = (self.counter as f64) / since_last_update.as_secs_f64();
            self.last_update = now;
            self.counter = 0;
            Some(self.fps)
        } else {
            None
        }
    }

    /// The FPS since last update.
    /// Returns `NaN` before the first update.
    pub fn fps(&self) -> f64 {
        self.fps
    }
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Application<'cx> {
    window: winit::window::Window,
    last_window_event: Instant,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    text: Line<'static, 'cx>,
    rect_floating: Rect<'cx>,
    rect_point: Rect<'cx>,
    context: &'cx Context,
}

impl<'cx> Application<'cx> {
    pub fn new(context: &'cx Context, window: winit::window::Window) -> Self {
        let scale_factor = window.scale_factor() as f32;
        Self {
            window,
            input_helper: InputHelper::new(),
            fps_counter: FpsCounter::new(),
            last_window_event: Instant::now(),
            text: {
                let mut line = Line::new(context);
                line.set_string("t = --.--".into());
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0.5, 0.5, 0.5, 0.5));
                line.set_font_size(20. * scale_factor);
                line
            },
            rect_floating: {
                let mut rect = Rect::new(context);
                rect.set_size(scale_factor * vec2(4., 4.));
                rect
            },
            rect_point: {
                let mut rect = Rect::new(context);
                rect.set_size(scale_factor * vec2(20., 20.));
                rect.uniform_fill(Color::new(1., 1., 1., 1.));
                rect
            },
            context,
        }
    }

    fn draw(&mut self) {
        let mut frame = self.context.display.draw();
        let frame_size = Vector2::from(frame.get_dimensions()).map(|x| x as f32);

        self.clear_frame(&mut frame);

        let seconds = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        let t = (seconds.sin() as f32 + 1.) / 2.;

        let ps = &[
            point2(100., frame_size.y / 2.),
            point2(frame_size.x - 100., 100.),
            point2(frame_size.x / 2., frame_size.y / 2. + 200.),
            point2(frame_size.x - 100., frame_size.y - 200.),
            point2(100., frame_size.y / 2.),
            point2(frame_size.x - 100., 100.),
        ];
        let p = bezier::lerp(ps, t);

        self.rect_floating
            .set_fill_color(Color::new(1. - t, 1., t, 1.).into());
        self.rect_floating.draw(&mut frame, p.sub_element_wise(2.));
        for p in ps {
            self.rect_point.draw(&mut frame, p.sub_element_wise(10.));
        }

        // self.text.set_string(format!("t = {t}").into());
        // self.text.draw(&mut frame, point2(10., 10.));

        frame.finish().unwrap();
        self.fps_counter.frame();
    }

    fn clear_frame(&mut self, frame: &mut glium::Frame) {
        let clear_color = (0.1, 0.1, 0.1, 1.);
        frame.clear_color_and_depth(clear_color, 1.);
    }

    #[allow(unused_variables)]
    fn before_window_event(&mut self, duration_since_last_window_event: Duration) {}

    #[allow(unused_variables)]
    fn key_down(&mut self, key_code: KeyCode, _text: Option<&str>, is_repeat: bool) {}

    #[allow(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}

    #[allow(unused_variables)]
    fn cursor_moved(&mut self, delta: Vector2<f32>) {}
}

impl winit::application::ApplicationHandler for Application<'_> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        _ = event_loop;
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let now = Instant::now();
        let duration_since_last_window_event = now.duration_since(self.last_window_event);
        self.last_window_event = now;
        self.before_window_event(duration_since_last_window_event);
        match event {
            winit::event::WindowEvent::CloseRequested => event_loop.exit(),
            winit::event::WindowEvent::RedrawRequested => {
                self.draw();
                self.window.request_redraw();
            }
            winit::event::WindowEvent::Resized(window_size) => {
                self.context
                    .resize(Vector2::new(window_size.width, window_size.height));
                self.window.request_redraw();
            }
            winit::event::WindowEvent::KeyboardInput {
                device_id: _,
                event,
                is_synthetic: _,
            } => {
                self.input_helper.update_key_event(&event);
                match event.physical_key {
                    PhysicalKey::Code(key_code) => {
                        if event.state.is_pressed() {
                            self.key_down(key_code, event.text.as_deref(), event.repeat)
                        } else {
                            self.key_up(key_code)
                        }
                    }
                    PhysicalKey::Unidentified(_) => (),
                }
            }
            _ => (),
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            winit::event::DeviceEvent::MouseMotion { delta } => {
                self.cursor_moved(Vector2::new(delta.0 as f32, delta.1 as f32));
            }
            winit::event::DeviceEvent::MouseWheel { delta: _ } => (),
            winit::event::DeviceEvent::Motion { axis: _, value: _ } => (),
            winit::event::DeviceEvent::Button {
                button: _,
                state: _,
            } => (),
            _ => (),
        }
    }
}
