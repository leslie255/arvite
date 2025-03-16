use font_types::{FWord, Fixed, LongDateTime};

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeadTable {
    pub(crate) version: Fixed,
    pub(crate) font_revision: Fixed,
    pub(crate) checksum_adjustment: u32,
    pub(crate) magic_number: u32,
    pub(crate) flags: u16,
    pub(crate) units_per_em: u16,
    pub(crate) created: LongDateTime,
    pub(crate) modified: LongDateTime,
    pub(crate) x_min: FWord,
    pub(crate) y_min: FWord,
    pub(crate) x_max: FWord,
    pub(crate) y_max: FWord,
    pub(crate) mac_style: u16,
    pub(crate) lowest_rec_ppem: u16,
    pub(crate) font_direction_hint: i16,
    pub(crate) index_to_loc_format: i16,
    pub(crate) glyph_data_format: i16,
}

impl_read_from!(
    HeadTable,
    version,
    font_revision,
    checksum_adjustment,
    magic_number,
    flags,
    units_per_em,
    created,
    modified,
    x_min,
    y_min,
    x_max,
    y_max,
    mac_style,
    lowest_rec_ppem,
    font_direction_hint,
    index_to_loc_format,
    glyph_data_format,
);
