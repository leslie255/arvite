#![feature(array_chunks, f16)]
#![allow(linker_messages)]

pub mod bind_group;
mod bmp;
pub mod buffer;
pub mod context;
pub mod pipeline;
pub mod shapes;
pub mod texture;
mod utils;

use bind_group::BindGroupBuilder;
use buffer::{IndexBuffer, Vertex2d, VertexBuffer};
use context::{Context, Surface};
use pipeline::{PipelineBuilder, WgslShaderSource};
use shapes::{Circle, ClassicText, Rectangle, TexturedRectangle};
use texture::Texture2d;

use cgmath::*;
use utils::{Wait, time};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

#[allow(dead_code)]
fn map_buffer(buffer_slice: wgpu::BufferSlice) {
    buffer_slice.map_async(wgpu::MapMode::Read, |result| result.unwrap());
}

/// `width * target_pixel_byte_cost` must be a multiple of `256`. This is required because `copy_texture_to_buffer` requires
/// texture with bytes per row of multiple of `256`.
/// Texture also cannot be in any compressed formats.
#[allow(dead_code)]
fn read_data_from_texture(
    context: &Context,
    width: u32,
    height: u32,
    texture: &wgpu::Texture,
) -> Vec<u8> {
    // `debug_assert` because `block_copy_size` would return `None` later anyways.
    debug_assert!(!texture.format().is_compressed());
    let bytes_per_pixel = texture.format().block_copy_size(None).unwrap();
    assert!((width * bytes_per_pixel) % 256 == 0);
    let mut encoder = context
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let buffer_size = bytes_per_pixel * width * height;
    let staging_buffer = context.wgpu_device.create_buffer(
        &(wgpu::BufferDescriptor {
            size: buffer_size.into(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        }),
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * bytes_per_pixel),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    let command_buffer = encoder.finish();
    context.wgpu_queue.submit([command_buffer]);

    let mut result_data = Vec::<u8>::with_capacity(buffer_size as usize);
    let buffer_slice = staging_buffer.slice(..);
    map_buffer(buffer_slice);
    context.wgpu_device.poll(wgpu::MaintainBase::Wait);
    let view = buffer_slice.get_mapped_range();
    result_data.extend_from_slice(&view[..]);
    drop(view);
    staging_buffer.unmap();

    result_data
}

const MANDELBROT_WGSL: WgslShaderSource<'_> = wgsl! {
    fn mandelbrot(c: vec2<f32>) -> f32 {
        let B: f32 = 256.0;
        var n: f32 = 0.0;
        var z: vec2<f32> = vec2<f32>(0.0, 0.0);
        for (var i: i32 = 0; i < 512; i = i + 1) {
            z = vec2<f32>(
                z.x * z.x - z.y * z.y,
                2.0 * z.x * z.y
            ) + c;
            if dot(z, z) > B * B {
                break;
            }
            n = n + 1.0;
        }
        return select(n, 0.0, n > 511.0);
    }

    struct VertexOutput {
        @location(0) uv: vec2<f32>,
        @builtin(position) position: vec4<f32>,
    };

    @vertex
    fn vs_main(@location(0) position: vec2<f32>) -> VertexOutput {
        var result: VertexOutput;
        result.uv = position;
        result.position = vec4<f32>(position.xy, 0.0, 1.0);
        return result;
    }

    @fragment
    fn fs_main(vertex: VertexOutput) -> @location(0) vec4<f32> {
        // Some simple scaling and movements.
        let c = vertex.uv * 1.2 - vec2<f32>(0.5, 0.0);
        let n = mandelbrot(c);
        return vec4<f32>(
            smoothstep(n, 0.0, 12.0),
            smoothstep(n, 0.0, 6.0),
            smoothstep(n, 0.0, 2.0),
            1.0,
        );
    }
};

struct Application<'cx, 'window> {
    window: Option<&'window Window>,
    window_surface: Surface<'window>,
    context: &'cx Context,
    /// If the current frame was the first frame after a resize.
    is_first_frame_after_resize: bool,
    frame_counter: u64,
    rectangle0: Option<TexturedRectangle<'cx>>,
    rectangle1: Option<TexturedRectangle<'cx>>,
}

impl<'cx, 'window> Application<'cx, 'window> {
    pub fn new(
        window: Option<&'window Window>,
        wgpu_surface: wgpu::Surface<'window>,
        context: &'cx Context,
    ) -> Self {
        Self {
            window,
            window_surface: Surface::for_window(
                window,
                wgpu_surface,
                wgpu::TextureFormat::Bgra8UnormSrgb,
            ),
            context,
            is_first_frame_after_resize: false,
            frame_counter: 0,
            rectangle0: None,
            rectangle1: None,
        }
    }

    fn request_redraw(&self) {
        if let Some(window) = self.window {
            window.request_redraw();
        }
    }

    fn draw_texture0(&self) -> Texture2d {
        let size = vec2(1024, 1024);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = Texture2d::drawable_bindable(self.context, size, format);
        let texture_view = texture.create_view();

        let surface = Surface::for_texture(texture_view);

        let mut encoder = self
            .context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut render_pass =
            surface.create_render_pass(&mut encoder, Some(wgpu::Color::WHITE), None, None, None);

        let mut circle = Circle::new(self.context, &surface)
            .with_fill_color(vec4(0.5, 0.5, 1.0, 1.0))
            .with_outer_radius(384.0)
            .with_inner_radius(256.0);
        circle.draw(&surface, &mut render_pass, Matrix4::identity());

        let mut draw_rectangle = |size: f32, color: Vector3<f32>| {
            let mut rectangle = Rectangle::new(self.context, &surface)
                .with_size(vec2(size, size))
                .with_fill_color(color.extend(1.0));
            rectangle.draw(&surface, &mut render_pass, Matrix4::identity());
            let mut text = ClassicText::new(self.context, &surface);
            let [r, g, b] = color.map(|x| (x * 255.0) as u8).into();
            *text.string_mut() = format!("{size}x{size}\n0x{r:02X}{g:02X}{b:02X}");
            text.set_fg_color(color.map(|x| 1.0 - x).extend(1.0));
            text.draw(
                &surface,
                &mut render_pass,
                Matrix4::from_translation(vec3(0.0, size, 0.0)),
            );
        };
        draw_rectangle(512.0, vec3(1.0, 1.0, 0.0));
        draw_rectangle(384.0, vec3(1.0, 0.0, 1.0));
        draw_rectangle(256.0, vec3(0.0, 1.0, 1.0));

        let mut text = ClassicText::new(self.context, &surface);
        *text.string_mut() = String::from("This\nIs\nA\nTexture");
        *text.text_height_mut() = 72.0;
        text.set_fg_color(vec4(0.0, 0.0, 0.0, 1.0));
        text.draw(
            &surface,
            &mut render_pass,
            Matrix4::from_translation(vec3(-512.0, 512.0, 0.0)),
        );

        drop(render_pass);

        self.context.wgpu_queue.submit([encoder.finish()]);

        // let texture_data =
        //     read_data_from_texture(self.context, size.x, size.y, &texture.wgpu_texture);
        // bmp::save_bmp("texture0.bmp", size, bmp::PixelFormat::Rgba8, &texture_data).unwrap();

        texture
    }

    fn draw_texture1(&self) -> Texture2d {
        let size = vec2(2048, 2048);
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let texture = Texture2d::drawable_bindable(self.context, size, format);
        let texture_view = texture.create_view();

        let surface = Surface::for_texture(texture_view);

        let mut encoder = self
            .context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut render_pass =
            surface.create_render_pass(&mut encoder, Some(wgpu::Color::WHITE), None, None, None);

        let shader = MANDELBROT_WGSL.compile(self.context);

        let vertex_buffer = VertexBuffer::new_initialized(self.context, &[
            Vertex2d::new([-1.0, -1.0]),
            Vertex2d::new([1.0, -1.0]),
            Vertex2d::new([1.0, 1.0]),
            Vertex2d::new([-1.0, 1.0]),
        ]);
        let index_buffer = IndexBuffer::<u16>::new_initialized(self.context, &[0, 1, 2, 2, 3, 0]);
        let render_pipeline = {
            let mut builder = PipelineBuilder::new(self.context, &shader);
            builder.texture_format(format);
            builder.add_vertex_buffer(&vertex_buffer);
            builder.build()
        };
        let bind_group = {
            let layout = render_pipeline.get_bind_group_layout(0);
            BindGroupBuilder::new(self.context, &layout).build()
        };
        render_pass.set_pipeline(&render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        vertex_buffer.set(&mut render_pass, 0, ..);
        index_buffer.set(&mut render_pass, ..);
        render_pass.draw_indexed(0..6, 0, 0..1);

        _ = &mut render_pass;

        drop(render_pass);

        self.context.wgpu_queue.submit([encoder.finish()]);

        // let texture_data =
        //     read_data_from_texture(self.context, size.x, size.y, &texture.wgpu_texture);
        // bmp::save_bmp("texture1.bmp", size, bmp::PixelFormat::Rgba8, &texture_data).unwrap();

        texture
    }

    fn draw(&mut self) {
        self.window_surface.begin_drawing();

        let mut encoder = self
            .context
            .wgpu_device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let mut render_pass = self.window_surface.create_render_pass(
            &mut encoder,
            Some(wgpu::Color::BLACK),
            None,
            None,
            None,
        );

        let scale_factor = self.window.map(|w| w.scale_factor() as f32).unwrap_or(1.0);

        let rectangle_size = vec2(256.0, 256.0) * scale_factor;

        if self.rectangle0.is_none() {
            let (duration, texture) = time(|| self.draw_texture0());
            println!(
                "[INFO] texture 0 took {} seconds to draw",
                duration.as_secs_f64()
            );
            let rectangle =
                TexturedRectangle::new(self.context, &self.window_surface, texture.create_view());
            self.rectangle0 = Some(rectangle);
        }
        let rectangle0 = self.rectangle0.as_mut().unwrap();
        *rectangle0.size_mut() = rectangle_size;
        let position = point2(-rectangle_size.x / 2.0, 0.0) - rectangle_size / 2.0;
        rectangle0.draw(
            &self.window_surface,
            &mut render_pass,
            Matrix4::from_translation(position.to_vec().extend(0.0)),
        );

        if self.rectangle1.is_none() {
            let (duration, texture) = time(|| self.draw_texture1());
            println!(
                "[INFO] texture 1 took {} seconds to draw",
                duration.as_secs_f64()
            );
            let rectangle =
                TexturedRectangle::new(self.context, &self.window_surface, texture.create_view());
            self.rectangle1 = Some(rectangle);
        }
        let rectangle1 = self.rectangle1.as_mut().unwrap();
        *rectangle1.size_mut() = rectangle_size;
        let position = point2(rectangle_size.x / 2.0, 0.0) - rectangle_size / 2.0;
        rectangle1.draw(
            &self.window_surface,
            &mut render_pass,
            Matrix4::from_translation(position.to_vec().extend(0.0)),
        );

        drop(render_pass);

        self.context.wgpu_queue.submit([encoder.finish()]);

        if let Some(window) = self.window {
            window.pre_present_notify();
        }

        self.window_surface.present()
    }
}

impl ApplicationHandler for Application<'_, '_> {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let (frame_time, ()) = time(|| {
                    self.draw();
                });
                self.is_first_frame_after_resize = false;
                let frame_time_seconds = frame_time.as_secs_f64();
                let fps = 1.0 / frame_time_seconds;
                self.frame_counter = self.frame_counter.wrapping_add(1);
                println!(
                    "[INFO] frame {} time: {frame_time_seconds}s ({fps:.2} fps)",
                    self.frame_counter
                );
            }
            WindowEvent::Resized(physical_size) => {
                self.is_first_frame_after_resize = true;
                let size = vec2(physical_size.width as f32, physical_size.height as f32);
                self.window_surface.window_resized(self.context, size);
            }
            WindowEvent::ScaleFactorChanged {
                scale_factor: _,
                inner_size_writer: _,
            } => {
                self.is_first_frame_after_resize = true;
                if self.frame_counter != 0 {
                    self.request_redraw();
                }
            }
            _ => (),
        }
    }
}

fn main() {
    let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
    let event_loop = EventLoop::new().unwrap();
    #[allow(deprecated)]
    let window = event_loop
        .create_window(Window::default_attributes().with_title("WGPU Test"))
        .unwrap();
    let (wgpu_surface, context) = Context::for_window(wgpu_instance, &window).wait();
    let mut application = Application::new(Some(&window), wgpu_surface, &context);
    event_loop.run_app(&mut application).unwrap();
}
