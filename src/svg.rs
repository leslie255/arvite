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
    pub fn command_q(&mut self, point1: Point2<f32>, point2: Point2<f32>) {
        self.spline.append_quadratic([point1, point2]);
    }

    /// Executes a `C` command.
    pub fn command_c(&mut self, point1: Point2<f32>, point2: Point2<f32>, point3: Point2<f32>) {
        self.spline.append_cubic([point1, point2, point3]);
    }

    /// Executes a `Z` command.
    pub fn command_z(&mut self) {
        let point = self.spline.segments()[self.previous_m][0];
        self.spline.append_linear(point);
    }

    pub fn parse_command(&mut self, source: &str) -> Result<(), ()> {
        let mut tokens = source.split_whitespace();
        loop {
            let Some(token) = tokens.next() else {
                break Ok(());
            };
            match token {
                "M" => {
                    self.command_m(point2(
                        tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                    ));
                }
                "L" => {
                    self.command_l(point2(
                        tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                    ));
                }
                "Q" => {
                    self.command_q(
                        point2(
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        ),
                        point2(
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        ),
                    );
                }
                "C" => {
                    self.command_c(
                        point2(
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        ),
                        point2(
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        ),
                        point2(
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                            tokens.next().ok_or(())?.parse().map_err(|_| ())?,
                        ),
                    );
                }
                "Z" => {
                    self.command_z();
                }
                _ => break Err(()),
            }
        }
    }
}
