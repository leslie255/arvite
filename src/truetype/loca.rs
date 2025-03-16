use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LocaTable {
    pub(crate) is_long: bool,
    pub(crate) offsets: BytesRef,
}

impl LocaTable {
    pub(crate) fn load(
        reader: &mut ByteReader,
        maxp_table: &MaxpTable,
        head_table: &HeadTable,
    ) -> Option<Self> {
        let n_glyphs = maxp_table.num_glyphs as usize;
        let is_long = match head_table.index_to_loc_format {
            0 => false,
            1 => true,
            _ => return None,
        };
        Some(Self {
            is_long,
            offsets: reader.read_bytes(n_glyphs * if is_long { 4 } else { 2 })?,
        })
    }
}
