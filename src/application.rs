use std::time::{Duration, Instant};

use cgmath::*;
use glium::{
    Surface,
    winit::{
        self,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{
    color::Color,
    context::Context,
    input::InputHelper,
    shapes::{BezierSplinePath, Circle, Path, PathDrawingMode},
    text::Line,
};

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
    epoch: Instant,
    context: &'cx Context,
    text: Line<'static, 'cx>,
    spline: BezierSplinePath<'static, 'cx>,
    fix_point_circles: Circle<'cx>,
    control_lines: Path<'cx>,
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
                line.set_string("FPS : ---.---".into());
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0.5, 0.5, 0.5, 0.5));
                line.set_font_size(20. * scale_factor);
                line
            },
            spline: {
                let mut spline = BezierSplinePath::new(context);
                spline.set_resolution(64);
                spline.set_draw_mode(PathDrawingMode::Line);
                spline.set_color0(Color::new(1., 1., 0., 1.));
                spline.set_color1(Color::new(1., 0., 1., 1.));
                spline
            },
            fix_point_circles: {
                let mut circle = Circle::new(context);
                circle.set_outer_radius(scale_factor * 4.);
                circle.set_inner_radius(scale_factor * 3.);
                circle.uniform_fill(Color::new(1., 1., 1., 1.));
                circle
            },
            control_lines: {
                let mut path = Path::new(context);
                path.set_draw_mode(PathDrawingMode::Line);
                path
            },
            epoch: Instant::now(),
            context,
        }
    }

    fn draw(&mut self) {
        let mut frame = self.context.display.draw();

        self.clear_frame(&mut frame);

        let position = point2::<f32>(200., 200.);

        // Knots / control points.
        for &point in self.spline.points() {
            let r = self.fix_point_circles.outer_radius();
            self.fix_point_circles
                .draw(&mut frame, point + position.to_vec() - vec2(r / 2., r / 2.));
            self.fix_point_circles
                .uniform_fill(Color::new(1., 1., 1., 1.));
        }
        // Control lines.
        for &[point0, point1] in self.spline.points().array_chunks::<2>() {
            self.control_lines.clear();
            self.control_lines
                .push_point(point0, Color::new(0.5, 0.5, 0.5, 1.));
            self.control_lines
                .push_point(point1, Color::new(0.5, 0.5, 0.5, 1.));
            self.control_lines.draw(&mut frame, position);
        }
        // The spline.
        self.spline.draw(&mut frame, position);

        self.text.draw(&mut frame, point2(10., 10.));

        frame.finish().unwrap();
        if let Some(fps) = self.fps_counter.frame() {
            self.text.set_string(format!("FPS: {fps:.3}").into());
        }
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

    #[allow(unused_variables)]
    fn resized(&mut self, frame_size: Vector2<f32>) {
        let scale_factor = self.window.scale_factor() as f32;
        // M 209.682922 441.830322
        // C 209.682922 441.830322 512.784424 235.796143 391.867676 72.83374
        // C 277.502899 -81.298584 250.148407 188.090942 175.742783 84.677979
        // C 114.818741 0.002441 65.932556 37.512939 32.682919 95.830322
        // C -23.292421 194.006958 209.682922 441.830322 209.682922 441.830322 Z
        let points = self.spline.points_mut();
        if points.is_empty() {
            #[allow(clippy::excessive_precision)]
            points.extend_from_slice(&[
                point2(209.682922, 441.830322),
                point2(209.682922, 441.830322),
                point2(512.784424, 235.796143),
                point2(391.867676, 72.83374),
                point2(391.867676, 72.83374),
                point2(277.502899, -81.298584),
                point2(250.148407, 188.090942),
                point2(175.742783, 84.677979),
                point2(175.742783, 84.677979),
                point2(114.818741, 0.002441),
                point2(65.932556, 37.512939),
                point2(32.682919, 95.830322),
                point2(32.682919, 95.830322),
                point2(-23.292421, 194.006958),
                point2(209.682922, 441.830322),
                point2(209.682922, 441.830322),
            ]);
        }
    }
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
                let window_size = Vector2::new(window_size.width, window_size.height);
                self.context.resize(window_size);
                self.resized(self.context.display_size());
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
