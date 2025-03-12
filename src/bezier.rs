use cgmath::*;

pub fn lerp2(p0: Point2<f32>, p1: Point2<f32>, t: f32) -> Point2<f32> {
    Point2::add_element_wise((1. - t) * p0, t * p1)
}

pub fn bezier(ps: &[Point2<f32>], t: f32) -> Point2<f32> {
    let n = ps.len() - 1;
    let n_f = n as f32;
    let mut coeff_acc = 1.0f32;
    let mut result = point2::<f32>(0., 0.);
    for (i, &p) in ps.iter().rev().enumerate() {
        let i_f = i as f32;
        // The binomial coefficient.
        let coeff = {
            if i != 0 {
                coeff_acc *= n_f - i_f + 1.;
                coeff_acc /= i_f;
            }
            coeff_acc
        };
        // The bernstein coefficient.
        let b = coeff * (1. - t).powi((n - i) as i32) * t.powi((i) as i32);
        result += b * p.to_vec();
    }
    result
}

pub fn bezier_cubic(ps: [Point2<f32>; 4], t: f32) -> Point2<f32> {
    (-t.powi(3) + 3. * t.powi(2) - 3. * t + 1.) * ps[0]
        + (3. * t.powi(3) - 6. * t.powi(2) + 3. * t) * ps[1].to_vec()
        + (-3. * t.powi(3) + 3. * t.powi(2)) * ps[2].to_vec()
        + (t.powi(3)) * ps[3].to_vec()
}
