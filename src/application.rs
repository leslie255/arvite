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

#[allow(clippy::excessive_precision)]
const BEZIER_SPLINE_SHAPE: &[[Point2<f32>; 3]] = &[
    [
        point2(131.20703, 426.07813),
        point2(218.78906, 452.0625),
        point2(296.6914, 435.08203),
    ],
    [
        point2(440.8828, 240.86328),
        point2(382.09766, 136.92188),
        point2(331.60547, 47.69922),
    ],
    [
        point2(247.67969, 71.640625),
        point2(209.75, 121.703125),
        point2(172.5625, 76.984375),
    ],
    [
        point2(104.92969, 49.05078),
        point2(46.15625, 136.41797),
        point2(-2.9960938, 222.85156),
    ],
];

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
    context: &'cx Context,
    window: winit::window::Window,
    last_window_event: Instant,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    epoch: Instant,
    selected_point: Option<(usize, usize)>,
    spline_position: Point2<f32>,
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
            selected_point: None,
            spline_position: point2(200., 200.),
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
                spline.set_color(Color::new(1., 0.4, 0.5, 1.));
                spline.segments_mut().extend_from_slice(BEZIER_SPLINE_SHAPE);
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

        self.spline.draw(&mut frame, self.spline_position);

        // Knots / control points.
        for (i, segment) in self.spline.segments().iter().enumerate() {
            let is_selected = self
                .selected_point
                .is_some_and(|(i_selected, _)| i == i_selected);
            let r = self.fix_point_circles.outer_radius();
            if is_selected {
                for (j, point) in segment.iter().enumerate() {
                    let is_selected = self.selected_point.unwrap().1 == j;
                    let inner_radius = if is_selected { 0.0f32 } else { r - 1. };
                    self.fix_point_circles.set_inner_radius(inner_radius);
                    self.fix_point_circles.draw(
                        &mut frame,
                        point + self.spline_position.to_vec() - vec2(r, r),
                    );
                }
            } else {
                let inner_radius = if is_selected { 0.0f32 } else { r - 1. };
                self.fix_point_circles.set_inner_radius(inner_radius);
                self.fix_point_circles.draw(
                    &mut frame,
                    segment[1] + self.spline_position.to_vec() - vec2(r, r),
                );
            }
        }
        // Control lines.
        if let Some(&[point0, point1, point2]) = self
            .selected_point
            .and_then(|i| self.spline.segments().get(i.0))
        {
            self.control_lines.clear();
            self.control_lines
                .push_point(point0, Color::new(0.5, 0.5, 0.5, 1.));
            self.control_lines
                .push_point(point1, Color::new(0.5, 0.5, 0.5, 1.));
            self.control_lines
                .push_point(point2, Color::new(0.5, 0.5, 0.5, 1.));
            self.control_lines.draw(&mut frame, self.spline_position);
        }

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
    fn key_down(&mut self, key_code: KeyCode, _text: Option<&str>, is_repeat: bool) {
        if key_code == KeyCode::KeyP && !is_repeat {
            println!("Shape:");
            for (i, segement) in self.spline.segments().iter().enumerate() {
                println!("segment_{i} : {segement:?}");
            }
        }
    }

    #[allow(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}

    fn mouse_down(&mut self, button: u32) {
        let Some(location) = self.input_helper.cursor_position_physical() else {
            return;
        };
        if button == 0 {
            let previous_selected_point = self.selected_point;
            self.selected_point = None;
            for (i, segment) in self.spline.segments().iter().enumerate() {
                for (j, point) in segment.iter().enumerate() {
                    let point = *point + self.spline_position.to_vec();
                    let allowed_distance =
                        if previous_selected_point.is_some_and(|prev| prev == (i, j)) {
                            self.fix_point_circles.outer_radius() * 2.
                        } else {
                            self.fix_point_circles.outer_radius()
                        };
                    let distance = location.distance(point);
                    if distance <= allowed_distance {
                        self.selected_point = Some((i, j));
                    }
                }
            }
        } else if button == 1 {
            if let Some(i_selected) = self.selected_point {
                let selected_point = self.spline.segments()[i_selected.0][i_selected.1];
                let distance = location.distance(selected_point + self.spline_position.to_vec());
                if distance > self.fix_point_circles.outer_radius() * 2. {
                    self.selected_point = None;
                }
            } else {
                let draw_mode = self.spline.draw_mode();
                self.spline.set_draw_mode(match draw_mode {
                    PathDrawingMode::Line => PathDrawingMode::Fill,
                    PathDrawingMode::Fill => PathDrawingMode::Line,
                });
            }
        }
    }

    #[allow(unused_variables)]
    fn mouse_up(&mut self, button: u32) {}

    #[allow(unused_variables)]
    fn cursor_moved(&mut self, delta: Vector2<f32>) {
        let physical_delta = delta * self.window.scale_factor() as f32;
        if self.input_helper.button_is_pressed(0) {
            let Some(location) = self.input_helper.cursor_position_physical() else {
                return;
            };
            let Some(i_selected) = self.selected_point else {
                return;
            };
            let segment = self.spline.segments_mut();
            if i_selected.1 == 1 {
                segment[i_selected.0][0] += physical_delta;
                segment[i_selected.0][1] += physical_delta;
                segment[i_selected.0][2] += physical_delta;
            } else {
                segment[i_selected.0][i_selected.1] += physical_delta;
            }
        }
    }

    #[allow(unused_variables)]
    fn resized(&mut self, frame_size: Vector2<f32>) {}
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
                self.input_helper.notify_key_event(&event);
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
            winit::event::WindowEvent::CursorMoved {
                device_id: _,
                position,
            } => {
                self.input_helper
                    .notify_cursor_moved(position, self.window.scale_factor());
            }
            winit::event::WindowEvent::CursorLeft { device_id: _ } => {
                self.input_helper.notify_cursor_left();
            }
            winit::event::WindowEvent::CursorEntered { device_id: _ } => {
                self.input_helper.notify_cursor_entered();
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
                self.cursor_moved(Vector2::from(delta).map(|f| f as f32));
            }
            winit::event::DeviceEvent::MouseWheel { delta: _ } => (),
            winit::event::DeviceEvent::Button { button, state } => {
                self.input_helper.notify_button_event(button, state);
                match state {
                    winit::event::ElementState::Pressed => self.mouse_down(button),
                    winit::event::ElementState::Released => self.mouse_up(button),
                }
            }
            _ => (),
        }
    }
}
