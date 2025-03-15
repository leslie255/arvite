use cgmath::*;

use crate::shapes::BezierSplinePath;

#[derive(Debug)]
pub struct SvgPathBuilder<'a, 'cx> {
    spline: &'a mut BezierSplinePath<'static, 'cx>,
    previous_m: usize,
}

impl<'a, 'cx> SvgPathBuilder<'a, 'cx> {
    pub fn new(spline: &'a mut BezierSplinePath<'static, 'cx>) -> Self {
        Self {
            spline,
            previous_m: 0,
        }
    }

    /// Executes an `M` command.
    pub fn command_m(&mut self, point: Point2<f32>) {
        if let Some(last_segment) = self.spline.segments_mut().last_mut() {
            last_segment[2] = point2(f32::NAN, f32::NAN);
        }
        self.spline.segments_mut().push([point, point, point]);
        self.previous_m = self.spline.segments().len() - 1;
    }

    /// Executes an `L` command.
    pub fn command_l(&mut self, point: Point2<f32>) {
        self.spline.append_linear(point);
    }

    /// Executes a `Q` command.
    pub fn command_q(&mut self, points: [Point2<f32>; 3]) {
        self.spline.append_cubic(points);
    }

    /// Executes a `C` command.
    pub fn command_c(&mut self, points: [Point2<f32>; 3]) {
        self.spline.append_cubic(points);
    }

    /// Executes a `Z` command.
    pub fn command_z(&mut self) {
        let point = self.spline.segments()[self.previous_m][0];
        self.spline.append_linear(point);
    }
}
