use std::time::{Duration, Instant};

/// Waits out a future blockingly.
pub trait Wait: Future {
    fn wait(self) -> Self::Output;
}

impl<F: Future> Wait for F {
    /// Waits out a future blockingly.
    fn wait(self) -> Self::Output {
        pollster::block_on(self)
    }
}

/// Quickly time an operation.
pub fn time<T>(f: impl FnOnce() -> T) -> (Duration, T) {
    let before = Instant::now();
    let result = f();
    let duration = Instant::now() - before;
    (duration, result)
}

#[macro_export]
macro_rules! time {
    ($stuff:expr $(,)?) => {{
        let (duration, result) = $crate::utils::time(|| $stuff);
        let seconds = duration.as_secs_f64();
        println!("[{}:{}] operation took {seconds} seconds", file!(), line!());
        result
    }};
}
