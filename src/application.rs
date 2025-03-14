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
    [point2(131., 452.), point2(218., 452.), point2(296., 452.)],
    [point2(440., 240.), point2(382., 136.), point2(331., 47.)],
    [point2(247., 71.), point2(209., 121.), point2(172., 76.)],
    [point2(104., 49.), point2(46., 136.), point2(0., 222.)],
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

/// Whether to show the control elements for the bezier spline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlElementsMode {
    /// Always show all control points and lines.
    All,
    /// Only show the center points for most segments.
    Minimal,
    /// No control elements.
    None,
}

#[derive(Debug)]
pub struct Application<'cx> {
    context: &'cx Context,
    window: winit::window::Window,
    last_window_event: Instant,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    epoch: Instant,
    control_elements_mode: ControlElementsMode,
    selected_point: Option<(usize, usize)>,
    spline_position: Point2<f32>,
    fps_text: Line<'static, 'cx>,
    position_text: Line<'static, 'cx>,
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
            epoch: Instant::now(),
            control_elements_mode: ControlElementsMode::Minimal,
            fps_text: {
                let mut line = Line::new(context);
                line.set_string("FPS : ---.---".into());
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0.5, 0.5, 0.5, 0.5));
                line.set_font_size(20. * scale_factor);
                line
            },
            position_text: {
                let mut line = Line::new(context);
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0.5, 0.5, 0.5, 0.5));
                line.set_font_size(12. * scale_factor);
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
            context,
        }
    }

    fn draw(&mut self) {
        let mut frame = self.context.display.draw();

        self.clear_frame(&mut frame);

        self.spline.draw(&mut frame, self.spline_position);

        // Knots / control points.
        if self.control_elements_mode != ControlElementsMode::None {
            for i in 0..self.spline.segments().len() {
                self.draw_controls_for_segment(&mut frame, i);
            }
        }

        // Position label.
        if let Some(selected_point) = self
            .selected_point
            .and_then(|i| self.spline.segments().get(i.0).map(|segment| segment[i.1]))
        {
            let scale_factor = self.window.scale_factor() as f32;
            self.position_text
                .set_string(format!("{:.3} {:.3}", selected_point.x, selected_point.y).into());
            self.position_text.draw(
                &mut frame,
                selected_point + vec2(8., 8.) * scale_factor + self.spline_position.to_vec(),
            );
        }

        self.fps_text.draw(&mut frame, point2(10., 10.));

        frame.finish().unwrap();
        if let Some(fps) = self.fps_counter.frame() {
            self.fps_text.set_string(format!("FPS: {fps:.3}").into());
        }
    }

    fn draw_controls_for_segment(&mut self, frame: &mut glium::Frame, i: usize) {
        let segment = self.spline.segments()[i];
        let is_selected = self
            .selected_point
            .is_some_and(|(i_selected, _)| i == i_selected);
        if is_selected || self.control_elements_mode == ControlElementsMode::All {
            for (j, &point) in segment.iter().enumerate() {
                let point_is_selected = self.selected_point.is_some_and(|ij| ij == (i, j));
                self.draw_point(frame, point, point_is_selected);
            }
            self.draw_control_line(frame, segment);
        } else {
            self.draw_point(frame, segment[1], false);
        }
    }

    fn draw_point(&mut self, frame: &mut glium::Frame, point: Point2<f32>, is_selected: bool) {
        let r = self.fix_point_circles.outer_radius();
        let inner_radius = if is_selected { 0.0f32 } else { r - 1. };
        self.fix_point_circles.set_inner_radius(inner_radius);
        self.fix_point_circles
            .draw(frame, point + self.spline_position.to_vec() - vec2(r, r));
    }

    fn draw_control_line(&mut self, frame: &mut glium::Frame, segment: [Point2<f32>; 3]) {
        self.control_lines.clear();
        self.control_lines
            .push_point(segment[0], Color::new(0.5, 0.5, 0.5, 1.));
        self.control_lines
            .push_point(segment[1], Color::new(0.5, 0.5, 0.5, 1.));
        self.control_lines
            .push_point(segment[2], Color::new(0.5, 0.5, 0.5, 1.));
        self.control_lines.draw(frame, self.spline_position);
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
        } else if key_code == KeyCode::KeyC && !is_repeat {
            self.control_elements_mode = match self.control_elements_mode {
                ControlElementsMode::All => ControlElementsMode::None,
                ControlElementsMode::Minimal => ControlElementsMode::All,
                ControlElementsMode::None => ControlElementsMode::Minimal,
            }
        } else if key_code == KeyCode::Delete || key_code == KeyCode::Backspace && !is_repeat {
            match &mut self.selected_point {
                Some((i, 1)) => {
                    self.spline.segments_mut().remove(*i);
                    if let Some(new_i) = i.checked_sub(1) {
                        *i = new_i;
                    } else if !self.spline.segments().is_empty() {
                        *i = 0;
                    } else {
                        self.selected_point = None;
                    }
                }
                Some((i, j)) => {
                    let segment = &mut self.spline.segments_mut()[*i];
                    segment[*j] = segment[1];
                }
                _ => (),
            }
        } else if key_code == KeyCode::KeyF && !is_repeat {
            let draw_mode = self.spline.draw_mode();
            self.spline.set_draw_mode(match draw_mode {
                PathDrawingMode::Line => PathDrawingMode::Fill,
                PathDrawingMode::Fill => PathDrawingMode::Line,
            });
        }
    }

    #[allow(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}

    fn left_click_down(&mut self, position_physical: Point2<f32>) {
        let previous_selected_point = self.selected_point;
        self.selected_point = None;
        for (i, segment) in self.spline.segments().iter().enumerate() {
            for j in [0, 2, 1] {
                let point = segment[j] + self.spline_position.to_vec();
                let r = self.fix_point_circles.outer_radius();
                let is_preivously_selected =
                    previous_selected_point.is_some_and(|prev| prev == (i, j));
                let allowed_distance = if is_preivously_selected { r * 2. } else { r };
                let distance = position_physical.distance(point);
                if distance <= allowed_distance {
                    self.selected_point = Some((i, j));
                }
            }
        }
    }

    fn right_click_down(&mut self, position_physical: Point2<f32>) {
        if let Some(i_selected) = self.selected_point {
            let selected_segment = &mut self.spline.segments_mut()[i_selected.0];
            let selected_point = selected_segment[i_selected.1];
            let distance =
                position_physical.distance(selected_point + self.spline_position.to_vec());
            let r = self.fix_point_circles.outer_radius();
            if distance > r * 2. {
                let new_segment_position = position_physical - self.spline_position.to_vec();
                self.spline
                    .segments_mut()
                    .insert(i_selected.0, [new_segment_position; 3]);
                self.selected_point = Some((i_selected.0, 1));
            } else {
                if selected_segment[0].distance2(selected_segment[1]) == 0. {
                    selected_segment[0] = selected_segment[1] - vec2(r * 8., 0.);
                }
                if selected_segment[2].distance2(selected_segment[1]) == 0. {
                    selected_segment[2] = selected_segment[1] + vec2(r * 8., 0.);
                }
            }
        } else {
            let new_segment_position = position_physical - self.spline_position.to_vec();
            self.spline.segments_mut().push([new_segment_position; 3]);
            self.selected_point = Some((self.spline.segments().len() - 1, 1));
        }
    }

    fn left_click_dragged(&mut self, _delta_physical: Vector2<f32>, position_physical: Point2<f32>) {
        let Some(i_selected) = self.selected_point else {
            return;
        };
        let alt_is_down = self.input_helper.key_is_down(KeyCode::AltLeft)
            || self.input_helper.key_is_down(KeyCode::AltRight);
        let shift_is_down = self.input_helper.key_is_down(KeyCode::ShiftLeft)
            || self.input_helper.key_is_down(KeyCode::ShiftRight);
        let selected_segment = &mut self.spline.segments_mut()[i_selected.0];
        if i_selected.1 == 1 {
            let v0 = selected_segment[0] - selected_segment[1];
            let v2 = selected_segment[2] - selected_segment[1];
            selected_segment[1] = position_physical - self.spline_position.to_vec();
            if !alt_is_down {
                selected_segment[0] = selected_segment[1] + v0;
                selected_segment[2] = selected_segment[1] + v2;
            }
        } else {
            selected_segment[i_selected.1] = position_physical - self.spline_position.to_vec();
            let selected_point = selected_segment[i_selected.1];
            if !alt_is_down {
                let center_point = selected_segment[1];
                let oppsite_point = &mut selected_segment[if i_selected.1 == 0 { 2 } else { 0 }];
                let d = if shift_is_down {
                    selected_point.distance(center_point)
                } else {
                    oppsite_point.distance(center_point)
                };
                let v = (selected_point - center_point).normalize_to(d);
                *oppsite_point = center_point - v;
            }
        }
    }

    fn mouse_down(&mut self, button: u32) {
        let Some(position_physical) = self.input_helper.cursor_position_physical() else {
            return;
        };
        if button == 0 {
            self.left_click_down(position_physical);
        } else if button == 1 {
            self.right_click_down(position_physical);
        }
    }

    #[allow(unused_variables)]
    fn mouse_up(&mut self, button: u32) {}

    #[allow(unused_variables)]
    fn cursor_moved(&mut self, delta: Vector2<f32>) {
        let delta_physical = delta * self.window.scale_factor() as f32;
        let position_physical = self.input_helper.cursor_position_physical();
        if self.input_helper.button_is_pressed(0) {
            if let Some(position_physical) = position_physical {
                self.left_click_dragged(delta_physical, position_physical);
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
