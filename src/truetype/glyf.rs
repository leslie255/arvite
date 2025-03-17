//! Manages reading the `glyf` table.

use cgmath::*;
use font_types::FWord;

use super::*;

/// Pedantically this isn't a term used in the actual TrueType manual.
/// It just refers to the shared part of simple and compound glyphs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct GlyphHeader {
    pub(crate) number_of_contours: i16,
    pub(crate) x_min: FWord,
    pub(crate) y_min: FWord,
    pub(crate) x_max: FWord,
    pub(crate) y_max: FWord,
}

impl_read_from!(GlyphHeader, number_of_contours, x_min, y_min, x_max, y_max);

impl GlyphHeader {
    /// Is this glyph a simple, as oppose to compound?
    /// This is signified by `number_of_contours` being negative.
    pub(crate) fn is_simple_glyph(self) -> bool {
        self.number_of_contours > 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SimpleGlyph {
    pub(crate) header: GlyphHeader,
    pub(crate) end_points: Vec<u16>,
    pub(crate) instruction_length: u16,
    pub(crate) instructions: BytesRef,
    pub(crate) points: Vec<GlyphPoint>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlyphPoint {
    pub(crate) position: Point2<i16>,
    pub(crate) is_on_curve: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Curve {
    Linear([Point2<i16>; 2]),
    Quadratic([Point2<i16>; 3]),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Contour {
    pub(crate) curves: Vec<Curve>,
}

fn one_cyclic<T: Copy>(iter: impl IntoIterator<Item = T>) -> impl Iterator<Item = T> {
    iterator! {
        let mut iter = iter.into_iter();
        let Some(first) = iter.next() else { return; };
        yield first;
        for item in iter {
            yield item;
        }
        yield first;
    }
}

#[test]
fn test() {
    use crate::dbg_unpretty;

    fn on_curve(x: i16, y: i16) -> GlyphPoint {
        GlyphPoint {
            position: point2(x, y),
            is_on_curve: true,
        }
    }
    fn off_curve(x: i16, y: i16) -> GlyphPoint {
        GlyphPoint {
            position: point2(x, y),
            is_on_curve: false,
        }
    }
    let contour = Contour::from_points([
        on_curve(30, 30),
        on_curve(40, 40),
        off_curve(50, 50),
        off_curve(60, 60),
        off_curve(70, 70),
    ]);
    for curve in &contour.curves {
        dbg_unpretty!(curve);
    }
}

impl Contour {
    pub(crate) fn from_points(points: impl IntoIterator<Item = GlyphPoint>) -> Self {
        let filtered_points = iterator! {
            let mut points = one_cyclic(points).peekable();
            while let Some(p0) = points.next() {
                let Some(&p1) = points.peek() else {
                    break;
                };
                yield p0;
                if !p0.is_on_curve && !p1.is_on_curve {
                    let p0_1 = point2(
                        (p0.position.x + p1.position.x) / 2,
                        (p0.position.y + p1.position.y) / 2,
                    );
                    yield GlyphPoint {
                        position: p0_1,
                        is_on_curve: true,
                    };
                }
            }
        };
        let mut points = one_cyclic(filtered_points).peekable();
        let mut curves = Vec::new();
        while let Some(p0) = points.next() {
            let Some(&p1) = points.peek() else {
                break;
            };
            if !p1.is_on_curve {
                points.next();
                let Some(&p2) = points.peek() else {
                    curves.push(Curve::Linear([p0.position, p1.position]));
                    break;
                };
                curves.push(Curve::Quadratic([p0.position, p1.position, p2.position]));
            } else {
                curves.push(Curve::Linear([p0.position, p1.position]));
            }
        }
        Self { curves }
    }
}

impl SimpleGlyph {
    pub(crate) fn contours(&self) -> impl Iterator<Item = Contour> {
        let end_points = self.end_points.iter().map(|&i| i as usize);
        let start_points =
            std::iter::once(0usize).chain(self.end_points.iter().map(|&i| i as usize + 1));
        let indices = start_points.zip(end_points);
        indices.map(|(i_start, i_end)| {
            Contour::from_points(self.points[i_start..=i_end].iter().copied())
        })
    }
}

impl ReadFrom for SimpleGlyph {
    /// Returns `None` if either error or is not a simple glyph.
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        let header = reader.read::<GlyphHeader>()?;
        if !header.is_simple_glyph() {
            return None;
        }
        let mut end_points = Vec::<u16>::with_capacity(2);
        for _ in 0..header.number_of_contours {
            end_points.push(reader.read::<u16>()?);
        }
        let n_points = *end_points.last()? as usize + 1;
        let instruction_length = reader.read::<u16>()?;
        let instructions = reader.read_bytes(instruction_length as usize)?;

        let flags = {
            let mut flags = Vec::<GlyphFlagByte>::with_capacity(n_points);
            let mut i_flag = 0;
            while i_flag < n_points {
                let flag = reader.read::<GlyphFlagByte>()?;
                flags.push(flag);
                i_flag += 1;
                if flag.repeat() {
                    let repeat_count = reader.read::<u8>()?;
                    i_flag += repeat_count as usize;
                    for _ in 0..repeat_count {
                        flags.push(flag);
                    }
                }
            }
            flags
        };

        let mut x_coordinates = Vec::<i16>::with_capacity(n_points);
        for flag in &flags {
            let previous_x = x_coordinates.last().copied().unwrap_or(0);
            let is_same_as_previous = !flag.x_short() && flag.x_is_same();
            let dx = if is_same_as_previous {
                0
            } else if flag.x_short() {
                let mut x = reader.read::<u8>()? as i16;
                if !flag.x_is_same() {
                    x = -x;
                }
                x
            } else {
                reader.read::<i16>()?
            };
            x_coordinates.push(previous_x.wrapping_add(dx));
        }

        let mut y_coordinates = Vec::<i16>::with_capacity(n_points);
        for flag in &flags {
            let previous_y = y_coordinates.last().copied().unwrap_or(0);
            let is_same_as_previous = !flag.y_short() && flag.y_is_same();
            let dy = if is_same_as_previous {
                0
            } else if flag.y_short() {
                let mut y = reader.read::<u8>()? as i16;
                if !flag.y_is_same() {
                    y = -y;
                }
                y
            } else {
                reader.read::<i16>()?
            };
            y_coordinates.push(previous_y.wrapping_add(dy));
        }

        let points: Vec<_> = flags
            .into_iter()
            .zip(x_coordinates)
            .zip(y_coordinates)
            .map(|((flags, x), y)| GlyphPoint {
                position: point2(x, y),
                is_on_curve: flags.on_curve(),
            })
            .collect();

        Some(SimpleGlyph {
            header,
            end_points,
            instruction_length,
            instructions,
            points,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct GlyphFlagByte(pub(crate) u8);

impl Debug for GlyphFlagByte {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GlyphFlagByte")
            .field("on_curve", &self.on_curve())
            .field("x_short", &self.x_short())
            .field("y_short", &self.y_short())
            .field("repeat", &self.repeat())
            .field("x_is_same", &self.x_is_same())
            .field("y_is_same", &self.y_is_same())
            .finish()
    }
}

impl GlyphFlagByte {
    pub(crate) const fn on_curve(self) -> bool {
        (self.0 & 0b0000_0001) != 0
    }

    pub(crate) const fn x_short(self) -> bool {
        (self.0 & 0b0000_0010) != 0
    }

    pub(crate) const fn y_short(self) -> bool {
        (self.0 & 0b0000_0100) != 0
    }

    pub(crate) const fn repeat(self) -> bool {
        (self.0 & 0b0000_1000) != 0
    }

    pub(crate) const fn x_is_same(self) -> bool {
        (self.0 & 0b0001_0000) != 0
    }

    pub(crate) const fn y_is_same(self) -> bool {
        (self.0 & 0b0010_0000) != 0
    }
}

impl ReadFrom for GlyphFlagByte {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self)
    }
}
