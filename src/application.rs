// This is very much speghetti code since I intended it to be a very simple testing code at first.

use std::{
    fmt::Debug,
    fs::File,
    mem,
    time::{Duration, Instant},
};

use cgmath::*;
use clipboard::{ClipboardContext, ClipboardProvider};
use glium::{
    Surface,
    winit::{
        self,
        event::MouseScrollDelta,
        keyboard::{KeyCode, PhysicalKey},
    },
};

use crate::{
    color::Color,
    context::Context,
    input::InputHelper,
    shapes::{BezierSplinePath, Path, PathDrawingMode, TestRect},
    svg::SvgPathBuilder,
    text::Line,
    truetype::{TrueTypeFont, glyf::Curve},
    utils::BoolToggle,
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

/// Manages scrolling and scaling of a canvas.
#[derive(Debug, Clone, Copy)]
pub struct Canvas {
    offset: Vector2<f32>,
    scale: f32,
    scale_factor: f32,
}

impl Canvas {
    pub fn new(scale_factor: f32) -> Self {
        Self {
            offset: vec2(0., 0.),
            scale: 1.,
            scale_factor,
        }
    }

    pub fn physical_scale(&self) -> f32 {
        self.scale * self.scale_factor
    }

    pub fn model(&self, position: Point2<f32>) -> Matrix4<f32> {
        let translation = self.offset + position.to_vec();
        Matrix4::from_scale(self.physical_scale())
            * Matrix4::from_translation(translation.extend(0.))
    }

    pub fn model_unscaled(&self, position: Point2<f32>, offset: Vector2<f32>) -> Matrix4<f32> {
        self.model(position + offset) * Matrix4::from_scale(1. / self.physical_scale())
    }

    pub fn inverse_model(&self) -> Matrix4<f32> {
        self.model(point2(0., 0.))
            .invert()
            .unwrap_or(Matrix4::identity())
    }

    /// Transform a point from screen-space back to in-canvas world space.
    /// `point` must be coordinate with `(0, 0)` at center of screen.
    pub fn inverse_transform(&self, point: Point2<f32>) -> Point2<f32> {
        let point3 = point3(point.x, point.y, 0.);
        let point_world = self.inverse_model().transform_point(point3);
        point2(point_world.x, point_world.y)
    }

    pub fn move_(&mut self, delta: Vector2<f32>) {
        self.offset += delta / self.physical_scale();
    }

    pub fn scale(&mut self, delta: f32) {
        let mut scale = self.scale.ln();
        scale += delta;
        self.scale = scale.exp().clamp(0.01, 100.);
    }
}

#[derive(Debug)]
pub struct CoordinateMarker<'cx> {
    cross: Path<'cx>,
    text: Line<'static, 'cx>,
    position: Point2<f32>,
}

impl<'cx> CoordinateMarker<'cx> {
    pub fn new(context: &'cx Context, position: Point2<f32>) -> Self {
        Self {
            cross: {
                let mut cross = Path::new(context);
                cross.push_point(point2(-4., -4.), Color::new(1., 1., 1., 1.));
                cross.push_point(point2(4., 4.), Color::new(1., 1., 1., 1.));
                cross.push_point(point2(f32::NAN, f32::NAN), Color::new(1., 1., 1., 1.));
                cross.push_point(point2(-4., 4.), Color::new(1., 1., 1., 1.));
                cross.push_point(point2(4., -4.), Color::new(1., 1., 1., 1.));
                cross.set_draw_mode(PathDrawingMode::Line);
                cross
            },
            text: {
                let string = format!("({:}, {:})", position.x, position.y);
                let mut line = Line::new_with_string(context, string.into());
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0., 0., 0., 0.));
                line.set_font_size(12.);
                line
            },
            position,
        }
    }

    pub fn context(&self) -> &'cx Context {
        self.cross.context()
    }

    pub fn draw(&mut self, frame: &mut glium::Frame, canvas: &Canvas) {
        self.cross.draw(
            frame,
            canvas.model_unscaled(self.position, vec2(0., 0.))
                * Matrix4::from_scale(canvas.scale_factor),
        );
        self.text.draw(
            frame,
            canvas.model_unscaled(self.position, vec2(0., 0.))
                * Matrix4::from_scale(canvas.scale_factor),
        );
    }
}

pub struct Application<'cx> {
    context: &'cx Context,
    window: winit::window::Window,
    clipboard_context: Option<ClipboardContext>,
    last_window_event: Instant,
    input_helper: InputHelper,
    fps_counter: FpsCounter,
    epoch: Instant,
    selected_point: Option<(usize, usize)>,
    show_fps: bool,
    canvas: Canvas,
    current_char: char,
    font: TrueTypeFont,
    fps_text: Line<'static, 'cx>,
    spline: BezierSplinePath<'static, 'cx>,
    test_rect: Option<TestRect<'cx>>,
    coordinate_markers: Vec<CoordinateMarker<'cx>>,
}

impl<'cx> Application<'cx> {
    pub fn new(context: &'cx Context, window: winit::window::Window) -> Self {
        let scale_factor = window.scale_factor() as f32;
        let font = {
            let file = context.ttf_font_file.try_clone().unwrap();
            TrueTypeFont::load_from_file(file).unwrap()
        };
        let mut self_ = Self {
            window,
            clipboard_context: match ClipboardProvider::new() {
                Ok(clipboard_context) => Some(clipboard_context),
                Err(e) => {
                    println!("[WARNING] unable to initialize clipboard context: {e:?}");
                    None
                }
            },
            input_helper: InputHelper::new(),
            fps_counter: FpsCounter::new(),
            last_window_event: Instant::now(),
            selected_point: None,
            show_fps: false,
            canvas: Canvas::new(scale_factor),
            epoch: Instant::now(),
            current_char: 'A',
            fps_text: {
                let mut line = Line::new(context);
                line.set_string("FPS : ---.---".into());
                line.set_fg_color(Color::new(1., 1., 1., 0.7));
                line.set_bg_color(Color::new(0.5, 0.5, 0.5, 0.5));
                line.set_font_size(16. * scale_factor);
                line
            },
            spline: {
                let mut spline = BezierSplinePath::new(context);
                spline.set_resolution(32);
                spline.set_is_closed(false);
                spline.set_draw_mode(PathDrawingMode::Line);
                spline.set_color(Color::new(1., 1., 1., 1.));
                spline
            },
            test_rect: None,
            font,
            coordinate_markers: Vec::new(),
            context,
        };
        self_.reset_coordinate_markers();
        self_.set_character('A');
        self_
    }

    fn convert_x(&self, x: i16) -> f32 {
        x as f32
    }

    fn convert_y(&self, y: i16) -> f32 {
        y.saturating_neg() as f32
    }

    fn convert_point(&self, point: Point2<i16>) -> Point2<f32> {
        point2(self.convert_x(point.x), self.convert_y(point.y))
    }

    fn set_character(&mut self, char: char) {
        let scale_factor = self.window.scale_factor() as f32;
        self.test_rect = Some(TestRect::new(self.context, &self.font, scale_factor, char));

        self.current_char = char;
        self.spline.segments_mut().clear();
        self.selected_point = None;
        let Some(glyph) = self.font.get_glyph(char as u32) else {
            return;
        };
        let mut spline = mem::replace(&mut self.spline, BezierSplinePath::new(self.context));
        let mut builder = SvgPathBuilder::new(&mut spline);
        for contour in glyph.contours() {
            let mut previous_point = point2(f32::NAN, f32::NAN);
            for &curve in &contour.curves {
                match curve {
                    Curve::Linear(points) => {
                        let [p0, p1] = points.map(|p| self.convert_point(p));
                        if previous_point != p0 {
                            builder.command_m(p0);
                        }
                        builder.command_l(p1);
                        previous_point = p1;
                    }
                    Curve::Quadratic(points) => {
                        let [p0, p1, p2] = points.map(|p| self.convert_point(p));
                        if previous_point != p0 {
                            builder.command_m(p0);
                        }
                        builder.command_q(p1, p2);
                        previous_point = p2;
                    }
                }
            }
        }
        self.spline = spline;
    }

    fn draw(&mut self) {
        let mut frame = self.context.display.draw();

        self.clear_frame(&mut frame);

        for marker in self.coordinate_markers.iter_mut() {
            marker.draw(&mut frame, &self.canvas);
        }

        // self.spline
        //     .draw(&mut frame, self.canvas.model(point2(0., 0.)));

        if self.show_fps {
            let (frame_width, frame_height) = frame.get_dimensions();
            self.fps_text.draw(
                &mut frame,
                Matrix4::from_translation(vec3(
                    -(frame_width as f32) / 2. + 10.,
                    -(frame_height as f32) / 2. + 10.,
                    0.,
                )),
            );
        }

        if let Some(test_rect) = self.test_rect.as_mut() {
            test_rect.draw(&mut frame, self.canvas.model(point2(0., -100.)));
        }

        frame.finish().unwrap();
        if let Some(fps) = self.fps_counter.frame() {
            self.fps_text.set_string(format!("FPS: {fps:.3}").into());
        }
    }

    fn clear_frame(&mut self, frame: &mut glium::Frame) {
        let clear_color = (0.1, 0.1, 0.1, 1.);
        frame.clear_color_and_depth(clear_color, 1.);
    }

    fn reset_coordinate_markers(&mut self) {
        self.coordinate_markers = vec![
            CoordinateMarker::new(self.context, point2(0., 0.)),
            CoordinateMarker::new(self.context, point2(100., 0.)),
        ];
    }

    fn scale_factor_changed(&mut self, scale_factor: f32) {
        self.fps_text.set_font_size(16. * scale_factor);
        self.canvas.scale_factor = scale_factor;
    }

    #[allow(unused_variables)]
    fn before_window_event(&mut self, duration_since_last_window_event: Duration) {}

    #[allow(unused_variables)]
    fn key_down(&mut self, key_code: KeyCode, text: Option<&str>, is_repeat: bool) {
        if !self.input_helper.control_is_down()
            && !self.input_helper.alt_is_down()
            && !self.input_helper.super_is_down()
            && !self.input_helper.key_is_down(KeyCode::Delete)
            && !self.input_helper.key_is_down(KeyCode::Backspace)
        {
            if let Some(text) = text {
                for char in text.chars() {
                    self.set_character(char);
                }
            }
        }
        match key_code {
            KeyCode::KeyP if !is_repeat && self.input_helper.control_is_down() => {
                println!("Dumping shape data");
                for (i, segement) in self.spline.segments().iter().enumerate() {
                    println!("segment_{i} : {segement:?}");
                }
            }
            KeyCode::KeyV if self.input_helper.control_is_down() => {
                println!("Pasted");
                let clipboard_content = self
                    .clipboard_context
                    .as_mut()
                    .and_then(|clipboard_context| clipboard_context.get_contents().ok());
                if let Some(string) = clipboard_content {
                    for char in string.chars() {
                        self.set_character(char);
                    }
                }
            }
            KeyCode::KeyL if !is_repeat && self.input_helper.control_is_down() => {
                println!("Toggled close path");
                self.spline.set_is_closed(!self.spline.is_closed());
            }
            KeyCode::Backspace if !is_repeat && self.input_helper.control_is_down() => {
                println!("Reset coordinate markers");
                self.reset_coordinate_markers();
            }
            KeyCode::KeyF if !is_repeat && self.input_helper.control_is_down() => {
                println!("Toggled showing FPS");
                self.show_fps.toggle();
            }
            KeyCode::Equal | KeyCode::NumpadAdd
                if self.input_helper.control_is_down() | self.input_helper.super_is_down() =>
            {
                self.canvas.scale(0.1);
            }
            KeyCode::Minus | KeyCode::Minus
                if self.input_helper.control_is_down() | self.input_helper.super_is_down() =>
            {
                self.canvas.scale(-0.1);
            }
            KeyCode::Digit0 | KeyCode::Numpad0
                if self.input_helper.control_is_down() | self.input_helper.super_is_down() =>
            {
                self.canvas.scale = 1.;
            }
            KeyCode::Minus | KeyCode::Minus
                if self.input_helper.control_is_down() | self.input_helper.super_is_down() =>
            {
                self.canvas.scale(-0.1);
            }
            KeyCode::Delete | KeyCode::Backspace => match &mut self.selected_point {
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
                    *j = 1;
                }
                _ => (),
            },
            KeyCode::KeyF if !is_repeat && self.input_helper.control_is_down() => {
                let draw_mode = self.spline.draw_mode();
                self.spline.set_draw_mode(match draw_mode {
                    PathDrawingMode::Line => PathDrawingMode::Fill,
                    PathDrawingMode::Fill => PathDrawingMode::Line,
                });
            }
            _ => (),
        }
    }

    #[allow(unused_variables)]
    fn key_up(&mut self, key_code: KeyCode) {}

    fn left_click_down(&mut self, _position_physical: Point2<f32>) {}

    fn right_click_down(&mut self, position_physical: Point2<f32>) {
        let position_centered = position_physical - self.context.display_size() / 2.;
        let position_world = self.canvas.inverse_transform(position_centered);
        self.coordinate_markers
            .push(CoordinateMarker::new(self.context, position_world));
    }

    fn left_click_dragged(
        &mut self,
        _delta_physical: Vector2<f32>,
        _position_physical: Point2<f32>,
    ) {
    }

    fn right_click_dragged(
        &mut self,
        _delta_physical: Vector2<f32>,
        _position_physical: Point2<f32>,
    ) {
    }

    fn middle_click_dragged(
        &mut self,
        delta_physical: Vector2<f32>,
        _position_physical: Point2<f32>,
    ) {
        self.canvas.move_(delta_physical);
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
        if let Some(position_physical) = self.input_helper.cursor_position_physical() {
            if self.input_helper.button_is_pressed(0) {
                self.left_click_dragged(delta_physical, position_physical);
            } else if self.input_helper.button_is_pressed(1) {
                self.right_click_dragged(delta_physical, position_physical);
            } else if self.input_helper.button_is_pressed(2) {
                self.middle_click_dragged(delta_physical, position_physical);
            }
        }
    }

    #[allow(unused_variables)]
    fn frame_resized(&mut self, frame_size: Vector2<f32>) {}

    fn scrolled(&mut self, delta_physical: Vector2<f32>) {
        if self.input_helper.alt_is_down() {
            self.canvas.scale(delta_physical.y / 120.);
        } else {
            self.canvas.move_(delta_physical);
        }
    }

    fn frame_size(&self) -> Vector2<f32> {
        self.context.display_size()
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
                self.frame_resized(self.context.display_size());
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
            winit::event::WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer: _,
            } => {
                self.scale_factor_changed(scale_factor as f32);
            }
            winit::event::WindowEvent::PinchGesture {
                device_id: _,
                delta,
                phase: _,
            } => {
                self.canvas.scale(delta as f32);
            }
            winit::event::WindowEvent::CursorLeft { device_id: _ } => {
                self.input_helper.notify_cursor_left();
            }
            winit::event::WindowEvent::CursorEntered { device_id: _ } => {
                self.input_helper.notify_cursor_entered();
            }
            winit::event::WindowEvent::DroppedFile(path) => {
                let file = match File::open(path) {
                    Ok(file) => file,
                    Err(e) => {
                        eprintln!("unable to open file: {e:?}");
                        return;
                    }
                };
                let font = match TrueTypeFont::load_from_file(file) {
                    Ok(font) => font,
                    Err(e) => {
                        eprintln!("unable to load file {e:?}");
                        return;
                    }
                };
                self.font = font;
                self.set_character(self.current_char);
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
            winit::event::DeviceEvent::MouseWheel { delta } => match delta {
                MouseScrollDelta::LineDelta(dx_lines, dy_lines) => {
                    self.scrolled(
                        vec2(dx_lines, dy_lines)
                            .map(|f| f * 16.0 * self.window.scale_factor() as f32),
                    );
                }
                MouseScrollDelta::PixelDelta(physical_position) => {
                    self.scrolled(vec2(physical_position.x, physical_position.y).map(|f| f as f32));
                }
            },
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
