use cgmath::*;

pub fn lerp2(ps: [Point2<f32>; 2], t: f32) -> Point2<f32> {
    Point2::add_element_wise((1. - t) * ps[0], t * ps[1])
}

pub fn lerp3(ps: [Point2<f32>; 3], t: f32) -> Point2<f32> {
    let p_0 = lerp2([ps[0], ps[1]], t);
    let p_1 = lerp2([ps[1], ps[2]], t);
    let ps_ = [p_0, p_1];
    lerp2(ps_, t)
}

pub fn lerp4(ps: [Point2<f32>; 3], t: f32) -> Point2<f32> {
    let p_0 = lerp2([ps[0], ps[1]], t);
    let p_1 = lerp2([ps[1], ps[2]], t);
    let ps_ = [p_0, p_1];
    lerp2(ps_, t)
}

/// Very inefficient bezier lerp.
pub fn lerp(ps: &[Point2<f32>], t: f32) -> Point2<f32> {
    match ps {
        [] => panic!(),
        &[p] => p,
        ps => {
            let p0 = lerp(&ps[1..], t);
            let p1 = lerp(&ps[..ps.len() - 1], t);
            lerp2([p0, p1], t)
        }
    }
}
