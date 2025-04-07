use std::{
    collections::HashMap,
    mem,
    ops::DerefMut as _,
    sync::{Arc, Mutex, MutexGuard, Weak},
};

use cgmath::*;

use crate::{shapes::ClassicFont, texture::TextureView2d};

#[derive(Debug)]
pub(crate) enum SurfaceBackend<'window> {
    Texture(wgpu::TextureView),
    WindowSurface {
        surface: wgpu::Surface<'window>,
        /// `None` for web canvas.
        window: Option<&'window winit::window::Window>,
        /// Set to `Some` when `begin_drawing` is called, `None` on `present`.
        surface_texture: Option<wgpu::SurfaceTexture>,
        /// Set to `Some` when `begin_drawing` is called, `None` on `present`.
        texture_view: Option<wgpu::TextureView>,
    },
}

impl<'window> SurfaceBackend<'window> {
    pub(crate) fn for_window(
        window: Option<&'window winit::window::Window>,
        wgpu_surface: wgpu::Surface<'window>,
    ) -> Self {
        Self::WindowSurface {
            surface: wgpu_surface,
            window,
            surface_texture: None,
            texture_view: None,
        }
    }

    /// Needs to be called before drawing, if surface is a window.
    pub(crate) fn begin_drawing(&mut self) {
        match self {
            SurfaceBackend::Texture(_) => (),
            SurfaceBackend::WindowSurface {
                surface,
                window: _,
                surface_texture,
                texture_view,
            } => {
                assert!(surface_texture.is_none() && texture_view.is_none());
                let surface_texture_ = surface.get_current_texture().unwrap();
                *texture_view = Some(
                    surface_texture_
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default()),
                );
                *surface_texture = Some(surface_texture_);
            }
        }
    }

    /// Needs to be called when finish drawing, if surface is a window.
    pub(crate) fn present(&mut self) {
        match self {
            SurfaceBackend::Texture(_) => (),
            SurfaceBackend::WindowSurface {
                surface: _,
                window,
                surface_texture,
                texture_view,
            } => {
                assert!(surface_texture.is_some());
                assert!(texture_view.is_some());
                if let Some(window) = *window {
                    window.pre_present_notify();
                }
                let surface_texture_ = surface_texture.take();
                surface_texture_.unwrap().present();
                *texture_view = None;
            }
        }
    }

    pub(crate) fn texture_view(&self) -> &wgpu::TextureView {
        match self {
            SurfaceBackend::Texture(texture_view) => texture_view,
            SurfaceBackend::WindowSurface {
                surface: _,
                window: _,
                surface_texture: _,
                texture_view,
            } => texture_view.as_ref().unwrap(),
        }
    }

    fn resized(&self, context: &Context, size: Vector2<f32>) {
        match self {
            SurfaceBackend::Texture(_) => (),
            SurfaceBackend::WindowSurface {
                surface,
                window: _,
                surface_texture: _,
                texture_view: _,
            } => {
                configure_surface(context, surface, size.map(|f| f as u32));
            }
        }
    }
}

/// A surface and its local data context.
#[derive(Debug)]
pub struct Surface<'window> {
    pub(crate) size: Vector2<f32>,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) backend: SurfaceBackend<'window>,
    pub(crate) pipeline_reuse_pool: RenderingPipelinePool,
}

impl<'window> Surface<'window> {
    pub fn for_texture(texture_view: TextureView2d) -> Self {
        Self {
            size: texture_view.size.map(|u| u as f32),
            format: texture_view.format,
            backend: SurfaceBackend::Texture(texture_view.wgpu_texture_view),
            pipeline_reuse_pool: RenderingPipelinePool::default(),
        }
    }

    pub(crate) fn for_window(
        window: Option<&'window winit::window::Window>,
        wgpu_surface: wgpu::Surface<'window>,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            size: {
                let size = window.map(|w| w.inner_size()).unwrap_or_default();
                vec2(size.width as f32, size.height as f32)
            },
            format,
            backend: SurfaceBackend::for_window(window, wgpu_surface),
            pipeline_reuse_pool: RenderingPipelinePool::default(),
        }
    }

    /// Needs to be called before drawing, if surface is a window.
    pub fn begin_drawing(&mut self) {
        self.backend.begin_drawing();
    }

    /// Needs to be called when finish drawing, if surface is a window.
    pub fn present(&mut self) {
        self.backend.present();
    }

    pub fn window_resized(&mut self, context: &Context, size: Vector2<f32>) {
        self.size = size;
        self.backend.resized(context, size);
    }

    pub fn create_render_pass<'encoder>(
        &self,
        encoder: &'encoder mut wgpu::CommandEncoder,
        clear_color: Option<wgpu::Color>,
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites>,
        occlusion_query_set: Option<&wgpu::QuerySet>,
    ) -> wgpu::RenderPass<'encoder> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[clear_color.map(|c| wgpu::RenderPassColorAttachment {
                view: self.backend.texture_view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(c),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment,
            timestamp_writes,
            occlusion_query_set,
        })
    }

    pub fn size(&self) -> Vector2<f32> {
        self.size
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn pipeline_reuse_pool(&self) -> &RenderingPipelinePool {
        &self.pipeline_reuse_pool
    }

    pub fn texture_view(&self) -> &wgpu::TextureView {
        self.backend.texture_view()
    }

    /// The orthographic projection matrix for surface of this size.
    /// This is used for converting from pixel-space coordinates to NDC space.
    pub fn projection_matrix(&self) -> Matrix4<f32> {
        cgmath::ortho(
            -self.size.x / 2.,
            self.size.x / 2.,
            -self.size.y / 2.,
            self.size.y / 2.,
            -1.,
            1.,
        )
    }
}

/// Stores a bunch of rendering pipelines for re-using.
#[derive(Debug, Default)]
pub struct RenderingPipelinePool {
    pub(crate) registries: Mutex<HashMap<&'static str, Weak<wgpu::RenderPipeline>>>,
}

impl RenderingPipelinePool {
    /// Sweep dead `Arc`s.
    /// Note that the `RenderPipeline`s drop function of a pipeline would be run when the last
    /// instance outside the pool is dropped. This function simply sweeps the heap allocation
    /// spaces.
    pub fn sweep(&self) {
        let mut registries = self.registries.lock().unwrap();
        let old_count = registries.len();
        let registries_ = mem::take(MutexGuard::deref_mut(&mut registries));
        *registries = registries_
            .into_iter()
            .filter(|(_, v)| v.strong_count() >= 1)
            .collect();
        let new_count = registries.len();
        let sweep_count = old_count - new_count;
        println!("[DEBUG] sweeped {sweep_count} allocations of `wgpu::RenderPipeline`s");
    }

    pub fn lookup_or_insert_with(
        &self,
        id: &'static str,
        pipeline: impl FnOnce() -> wgpu::RenderPipeline,
    ) -> Arc<wgpu::RenderPipeline> {
        let mut registries = self.registries.lock().unwrap();
        match registries.get(id).and_then(Weak::upgrade) {
            Some(pipeline) => pipeline,
            None => {
                println!("[DEBUG] creating new render pipeline with ID {id:?}");
                let pipeline = Arc::new(pipeline());
                registries.insert(id, Arc::downgrade(&pipeline));
                drop(registries);
                pipeline
            }
        }
    }
}

/// The global data context.
#[derive(Debug)]
pub struct Context {
    #[expect(dead_code)]
    pub(crate) wgpu_instance: wgpu::Instance,
    pub(crate) wgpu_device: wgpu::Device,
    pub(crate) wgpu_queue: wgpu::Queue,
    pub(crate) wgpu_adapter: wgpu::Adapter,
    pub(crate) big_blue_terminal_font: Mutex<Weak<ClassicFont>>,
}

impl Context {
    pub async fn for_window(
        wgpu_instance: wgpu::Instance,
        window: &winit::window::Window,
    ) -> (wgpu::Surface, Self) {
        let wgpu_surface = wgpu_instance.create_surface(window).unwrap();
        let surface_size = {
            let size = window.inner_size();
            vec2(size.width, size.height)
        };
        let self_ = Self::for_surface(wgpu_instance, surface_size, &wgpu_surface).await;
        (wgpu_surface, self_)
    }

    pub async fn for_surface(
        wgpu_instance: wgpu::Instance,
        surface_size: Vector2<u32>,
        wgpu_surface: &wgpu::Surface<'_>,
    ) -> Self {
        let power_preference = wgpu::PowerPreference::from_env()
            .inspect(|p| println!("[INFO] using WGPU power preference `{p:?}`"))
            .unwrap_or_default();
        let adapter = wgpu_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: Some(wgpu_surface),
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .unwrap();
        let self_ = Self::new(wgpu_instance, device, queue, adapter).await;
        configure_surface(&self_, wgpu_surface, surface_size);
        self_
    }

    pub async fn new(
        wgpu_instance: wgpu::Instance,
        wgpu_device: wgpu::Device,
        wgpu_queue: wgpu::Queue,
        wgpu_adapter: wgpu::Adapter,
    ) -> Self {
        Self {
            wgpu_instance,
            wgpu_device,
            wgpu_queue,
            wgpu_adapter,
            big_blue_terminal_font: Mutex::new(Weak::new()),
        }
    }

    /// The default `classic` font, Big Blue Terminal.
    pub(crate) fn big_blue_terminal_font(&self) -> Arc<ClassicFont> {
        let mut weak = self.big_blue_terminal_font.lock().unwrap();
        match weak.upgrade() {
            Some(x) => x,
            None => {
                let font = Arc::new(ClassicFont::big_blue_terminal(self));
                *weak = Arc::downgrade(&font);
                font
            }
        }
    }
}

pub fn configure_surface(context: &Context, surface: &wgpu::Surface<'_>, size: Vector2<u32>) {
    let config = surface
        .get_default_config(&context.wgpu_adapter, size.x, size.y)
        .unwrap();
    surface.configure(&context.wgpu_device, &config);
}
