use cgmath::*;

use glium::winit::{
    dpi::{LogicalPosition, PhysicalPosition},
    event::KeyEvent,
    keyboard::{KeyCode, PhysicalKey},
};

/// Keeps track of which keys are currently down.
#[derive(Debug, Clone)]
pub struct InputHelper {
    downed_keys: Vec<bool>,
    cursor_position_physical: Option<PhysicalPosition<f64>>,
    cursor_position_logical: Option<LogicalPosition<f64>>,
    downed_buttons: Vec<bool>,
}

impl InputHelper {
    pub fn new() -> Self {
        Self {
            downed_keys: vec![false; 256],
            cursor_position_physical: None,
            cursor_position_logical: None,
            downed_buttons: vec![false; 64],
        }
    }

    fn index_for_key(key_code: KeyCode) -> usize {
        // This is technically unsafe lol due to KeyCode not being a stable API, but like nah.
        (key_code as u8).into()
    }

    pub fn key_is_down(&self, key_code: KeyCode) -> bool {
        self.downed_keys[Self::index_for_key(key_code)]
    }

    pub fn button_is_pressed(&self, button: u32) -> bool {
        self.downed_buttons[button as usize]
    }

    pub fn cursor_position_physical(&mut self) -> Option<Point2<f32>> {
        self.cursor_position_physical
            .map(|p| point2(p.x as f32, p.y as f32))
    }

    pub fn cursor_position_logical(&mut self) -> Option<Point2<f32>> {
        self.cursor_position_logical
            .map(|p| point2(p.x as f32, p.y as f32))
    }

    pub fn notify_key_event(&mut self, key_event: &KeyEvent) {
        if key_event.repeat {
            return;
        }
        let key_code = match key_event.physical_key {
            PhysicalKey::Code(key_code) => key_code,
            PhysicalKey::Unidentified(_) => return,
        };
        let index = Self::index_for_key(key_code);
        self.downed_keys[index] = key_event.state.is_pressed();
    }

    pub fn notify_cursor_moved(&mut self, position: PhysicalPosition<f64>, scale_factor: f64) {
        self.cursor_position_physical = Some(position);
        self.cursor_position_logical = Some(position.to_logical(scale_factor));
    }

    pub fn notify_cursor_left(&mut self) {
        self.cursor_position_physical = None;
        self.cursor_position_logical = None;
    }

    pub fn notify_cursor_entered(&mut self) {}

    pub fn notify_button_event(&mut self, button: u32, state: glium::winit::event::ElementState) {
        self.downed_buttons[button as usize] = state.is_pressed();
    }

    pub fn shift_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ShiftLeft)
            || self.key_is_down(KeyCode::ShiftRight)
    }

    pub fn control_is_down(&self) -> bool {
        self.key_is_down(KeyCode::ControlLeft)
            || self.key_is_down(KeyCode::ControlRight)
    }

    pub fn alt_is_down(&self) -> bool {
        self.key_is_down(KeyCode::AltLeft)
            || self.key_is_down(KeyCode::AltRight)
    }

    pub fn super_is_down(&self) -> bool {
        self.key_is_down(KeyCode::SuperLeft)
            || self.key_is_down(KeyCode::SuperRight)
    }
}

impl Default for InputHelper {
    fn default() -> Self {
        Self::new()
    }
}
