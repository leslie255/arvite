#![allow(dead_code, linker_messages)]

use glium::{backend::glutin, winit};

pub mod application;
pub mod color;
pub mod input;
pub mod mesh;
pub mod rect;
pub mod resource;
pub mod text;
pub mod utils;
pub mod bezier;

use application::{Application, Context};

fn main() {
    unsafe {
        utils::this_thread_is_main_thread_pinky_promise();
    }

    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let (window, display) = glutin::SimpleWindowBuilder::new()
        .with_title("Bezier Curve Demo")
        .with_inner_size(800, 480)
        .build(&event_loop);
    let scale_factor = window.scale_factor();
    println!("[INFO] UI scale factor: {scale_factor}");
    if scale_factor != 1. {
        let _ = window.request_inner_size(winit::dpi::LogicalSize::new(800, 480));
    }

    let context = Context::load(display);

    let mut game = Application::new(&context, window);
    event_loop.run_app(&mut game).unwrap();
}
