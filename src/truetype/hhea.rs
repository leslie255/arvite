use super::*;

use font_types::{FWord, Fixed, UfWord};

#[derive(Debug, Clone, Copy)]
pub(crate) struct HheaTable {
    /// 0x00010000 (1.0)
    pub(crate) version: Fixed,
    /// Distance from baseline of highest ascender.
    pub(crate) ascent: FWord,
    /// Distance from baseline of lowest descender.
    pub(crate) descent: FWord,
    /// Typographic line gap.
    pub(crate) line_gap: FWord,
    /// Must be consistent with horizontal metrics.
    pub(crate) advance_width_max: UfWord,
    /// Must be consistent with horizontal metrics.
    pub(crate) min_left_side_bearing: FWord,
    /// Must be consistent with horizontal metrics.
    pub(crate) min_right_side_bearing: FWord,
    /// max(lsb + (x_max-x_min)).
    pub(crate) x_max_extent: FWord,
    /// used to calculate the slope of the caret (rise/run) set to 1 for vertical caret.
    pub(crate) caret_slope_rise: i16,
    /// 0 for vertical.
    pub(crate) caret_slope_run: i16,
    /// set value to 0 for non-slanted fonts.
    pub(crate) caret_offset: FWord,
    pub(crate) reserved: [i16; 4],
    /// 0 for current format.
    pub(crate) metric_data_format: i16,
    /// number of advance widths in metrics table.
    pub(crate) num_of_long_hor_metrics: u16,
}

impl_read_from!(
    HheaTable,
    version,
    ascent,
    descent,
    line_gap,
    advance_width_max,
    min_left_side_bearing,
    min_right_side_bearing,
    x_max_extent,
    caret_slope_rise,
    caret_slope_run,
    caret_offset,
    reserved,
    metric_data_format,
    num_of_long_hor_metrics,
);
