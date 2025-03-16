#![feature(
    iter_next_chunk,
    array_chunks,
    array_windows,
    iter_array_chunks,
    coroutines,
    coroutine_trait
)]
#![allow(dead_code, linker_messages)]

use context::Context;
use glium::{backend::glutin, winit};

pub mod bezier;
pub mod color;
pub mod context;
pub mod mesh;
pub mod shapes;
pub mod text;
pub mod truetype;

pub(crate) mod application;
pub(crate) mod input;
pub(crate) mod resource;
pub(crate) mod svg;
pub(crate) mod utils;

use application::Application;

fn main() {
    unsafe {
        utils::this_thread_is_main_thread_pinky_promise();
    }

    let event_loop = winit::event_loop::EventLoop::builder().build().unwrap();

    let (window, display) = glutin::SimpleWindowBuilder::new()
        .with_title("Font Render Test")
        .with_inner_size(800, 480)
        .build(&event_loop);
    let scale_factor = window.scale_factor();
    println!("[INFO] UI scale factor: {scale_factor}");
    if scale_factor != 1. {
        let _ = window.request_inner_size(winit::dpi::LogicalSize::new(800, 480));
    }

    let context = Context::new(display);

    let mut game = Application::new(&context, window);
    event_loop.run_app(&mut game).unwrap();
}
