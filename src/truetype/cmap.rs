//! Manages reading the `cmap` table.

use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CmapIndex {
    version: u16,
    number_subtables: u16,
}

impl_read_from!(CmapIndex, version, number_subtables);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CmapSubtableHeader {
    pub(crate) platform_id: u16,
    pub(crate) platform_specific_id: u16,
    pub(crate) offset: u32,
}

impl_read_from!(
    CmapSubtableHeader,
    platform_id,
    platform_specific_id,
    offset
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CmapTable {
    pub(crate) index: CmapIndex,
    pub(crate) subtables: Vec<CmapSubtableHeader>,
    pub(crate) format0_tables: Vec<CmapSubtableFormat0>,
    pub(crate) format4_tables: Vec<CmapSubtableFormat4>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CmapSubtableFormat0 {
    pub(crate) format: u16,
    pub(crate) length: u16,
    pub(crate) language: u16,
    pub(crate) glyph_index_array: FixedBytesRef<256>,
}

impl_read_from!(
    CmapSubtableFormat0,
    format,
    length,
    language,
    glyph_index_array,
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CmapSubtableFormat4 {
    pub(crate) format: u16,
    pub(crate) length: u16,
    pub(crate) language: u16,
    pub(crate) seg_count_x2: u16,
    pub(crate) search_range: u16,
    pub(crate) entry_selector: u16,
    pub(crate) range_shift: u16,
    pub(crate) end_code: BytesRef,
    pub(crate) reserve_pad: u16,
    pub(crate) start_code: BytesRef,
    pub(crate) id_delta: BytesRef,
    pub(crate) id_range_offset: BytesRef,
}

impl CmapSubtableFormat4 {
    pub(crate) fn segments(&self, reader: &ByteReader) -> impl Iterator<Item = CmapFormat4Segment> {
        iterator! {
            let mut start_codes = reader.subreader(self.start_code);
            let mut end_codes = reader.subreader(self.end_code);
            let mut id_deltas = reader.subreader(self.id_delta);
            let mut id_range_offsets = reader.subreader(self.id_range_offset);
            for _ in 0..(self.seg_count_x2 / 2) {
                let start_code = start_codes.read::<u16>().unwrap();
                let end_code = end_codes.read::<u16>().unwrap();
                let id_delta = id_deltas.read::<i16>().unwrap();
                let id_range_offset = id_range_offsets.read::<u16>().unwrap();
                yield CmapFormat4Segment { start_code, end_code, id_delta, id_range_offset };
            }
        }
    }

    pub(crate) fn search_for_codepoint(&self, codepoint: u16, reader: &ByteReader) -> u16 {
        // For now we'll just do a linear search.
        for segment in self.segments(reader) {
            if (segment.start_code..segment.end_code).contains(&codepoint) {
                let mut glyph_index = codepoint.wrapping_add_signed(segment.id_delta);
                glyph_index += segment.id_range_offset;
                return glyph_index;
            }
        }
        0
    }
}

impl ReadFrom for CmapSubtableFormat4 {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        let format: u16 = reader.read()?;
        let length: u16 = reader.read()?;
        let language: u16 = reader.read()?;
        let seg_count_x2: u16 = reader.read()?;
        Some(Self {
            format,
            length,
            language,
            seg_count_x2,
            search_range: reader.read()?,
            entry_selector: reader.read()?,
            range_shift: reader.read()?,
            end_code: reader.read_bytes(seg_count_x2 as usize)?,
            reserve_pad: reader.read()?,
            start_code: reader.read_bytes(seg_count_x2 as usize)?,
            id_delta: reader.read_bytes(seg_count_x2 as usize)?,
            id_range_offset: reader.read_bytes(seg_count_x2 as usize)?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CmapFormat4Segment {
    pub(crate) start_code: u16,
    pub(crate) end_code: u16,
    pub(crate) id_delta: i16,
    pub(crate) id_range_offset: u16,
}

impl CmapTable {
    /// `reader` should be a sub reader in the range of the cmap.
    pub(crate) fn load(reader: &mut ByteReader) -> Result<Self, TTFontLoadError> {
        let index = reader
            .read::<CmapIndex>()
            .ok_or(TTFontLoadError::MalformedCmap)?;
        let subtable_headers: Vec<CmapSubtableHeader> = reader
            .read_multiple(index.number_subtables as usize)
            .collect::<Vec<CmapSubtableHeader>>();
        if subtable_headers.len() != index.number_subtables as usize {
            return Err(TTFontLoadError::MalformedCmap);
        }
        let mut format0_tables = Vec::<CmapSubtableFormat0>::new();
        let mut format4_tables = Vec::<CmapSubtableFormat4>::new();
        for subtable_header in &subtable_headers {
            if subtable_header.platform_id != 0 {
                continue;
            }
            reader.goto(subtable_header.offset as usize);
            let format = reader.peek::<u16>().unwrap();
            match format {
                0 => format0_tables.push(reader.read::<CmapSubtableFormat0>().unwrap()),
                4 => format4_tables.push(reader.read::<CmapSubtableFormat4>().unwrap()),
                12 => {
                    println!("[WARNING] unsuppported cmap format 12");
                }
                _ => println!("[WARNING] unsupported cmap format {format}"),
            }
        }
        Ok(CmapTable {
            index,
            subtables: subtable_headers,
            format0_tables,
            format4_tables,
        })
    }
}
