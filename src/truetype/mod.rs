// Reference: https://developer.apple.com/fonts/TrueType-Reference-Manual/

pub(crate) mod cmap;
pub(crate) mod glyf;
pub(crate) mod head;
pub(crate) mod loca;
pub(crate) mod maxp;
pub(crate) mod reader;

use cmap::*;
use glyf::*;
use head::*;
use loca::*;
use maxp::*;

use reader::*;

use std::{
    collections::BTreeMap,
    fmt::{self, Debug},
    fs::File,
    io::{self, BufReader, Read},
};

use crate::{iterator, impl_read_from};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OffsetSubtable {
    pub(crate) scaler_type: u32,
    pub(crate) num_tables: u16,
    pub(crate) search_range: u16,
    pub(crate) entry_selector: u16,
    pub(crate) range_shift: u16,
}

impl_read_from!(
    OffsetSubtable,
    scaler_type,
    num_tables,
    search_range,
    entry_selector,
    range_shift
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableDirectory {
    pub(crate) tag: TableHeader,
    pub(crate) checksum: u32,
    pub(crate) offset: u32,
    pub(crate) length: u32,
}

impl_read_from!(TableDirectory, tag, checksum, offset, length);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From)]
pub(crate) struct TableHeader(pub [u8; 4]);

impl ReadFrom for TableHeader {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        Some(Self(reader.read_array()?))
    }
}

impl TableHeader {
    pub fn as_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }

    pub fn as_mut_str(&mut self) -> Option<&mut str> {
        std::str::from_utf8_mut(&mut self.0).ok()
    }

    pub fn from_str(str: &str) -> Option<Self> {
        let bytes = str.as_bytes();
        bytes.try_into().ok().map(Self)
    }
}

impl PartialEq<&str> for TableHeader {
    fn eq(&self, other: &&str) -> bool {
        other.as_bytes() == self.0
    }
}

impl Debug for TableHeader {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(s) = self.as_str() {
            Debug::fmt(&s, f)
        } else {
            Debug::fmt(&self.0, f)
        }
    }
}

#[derive(Debug)]
pub enum TTFontLoadError {
    MalformedHeader,
    IoError(io::Error),
    TableOutOfRange(String),
    MissingRequiredTable(String),
    MalformedCmap,
    UnsupportedFeature,
    MalformedHeadTable,
    BitmapFontIsNotSupported,
    MalformedLoca,
}

#[derive(Clone)]
pub struct TrueTypeFont {
    pub(crate) offset_subtable: OffsetSubtable,
    pub(crate) table_directories: BTreeMap<TableHeader, TableDirectory>,
    pub(crate) data: Box<[u8]>,
    pub(crate) head_table: Option<HeadTable>,
    pub(crate) maxp_table: Option<MaxpTable>,
    pub(crate) cmap_table: Option<CmapTable>,
    pub(crate) loca_table: Option<LocaTable>,
}

impl Debug for TrueTypeFont {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TrueTypeFont")
            .field("offset_subtable", &self.offset_subtable)
            .field("table_directories", &self.table_directories)
            .finish_non_exhaustive()
    }
}

impl TrueTypeFont {
    pub fn load_from_file(file: File) -> Result<Self, TTFontLoadError> {
        let bytes = {
            let mut bytes = Vec::new();
            let mut reader = BufReader::new(file);
            reader.read_to_end(&mut bytes).unwrap();
            bytes
        };
        Self::load_from_bytes(bytes.into())
    }

    pub fn load_from_bytes(bytes: Box<[u8]>) -> Result<Self, TTFontLoadError> {
        let mut reader = ByteReader::new(&bytes);
        let offset_subtable = reader
            .read::<OffsetSubtable>()
            .ok_or(TTFontLoadError::MalformedHeader)?;
        let table_directories: BTreeMap<TableHeader, TableDirectory> = reader
            .read_multiple::<TableDirectory>(offset_subtable.num_tables as usize)
            .map(|table_directory| (table_directory.tag, table_directory))
            .collect();
        if table_directories.len() != offset_subtable.num_tables as usize {
            return Err(TTFontLoadError::MalformedHeader);
        }
        let mut font = TrueTypeFont {
            offset_subtable,
            table_directories,
            data: bytes,
            head_table: None,
            maxp_table: None,
            cmap_table: None,
            loca_table: None,
        };
        font.load_head()?;
        font.load_maxp()?;
        font.load_cmap()?;
        font.load_loca()?;
        Ok(font)
    }

    pub(crate) fn load_cmap(&mut self) -> Result<(), TTFontLoadError> {
        let cmap_data = self
            .get_table_data(b"cmap")
            .ok_or(TTFontLoadError::MissingRequiredTable("cmap".into()))?;
        let mut reader = ByteReader::new(&self.data).subreader(cmap_data);
        self.cmap_table = Some(CmapTable::load(&mut reader)?);
        Ok(())
    }

    pub(crate) fn load_head(&mut self) -> Result<(), TTFontLoadError> {
        let head_data = self
            .get_table_data(b"head")
            .ok_or(TTFontLoadError::BitmapFontIsNotSupported)?;
        let mut reader = ByteReader::new(&self.data).subreader(head_data);
        self.head_table = Some(reader.read().ok_or(TTFontLoadError::MalformedHeadTable)?);
        Ok(())
    }

    pub(crate) fn load_maxp(&mut self) -> Result<(), TTFontLoadError> {
        let maxp_data = self
            .get_table_data(b"maxp")
            .ok_or(TTFontLoadError::MissingRequiredTable("maxp".into()))?;
        let mut reader = ByteReader::new(&self.data).subreader(maxp_data);
        self.maxp_table = Some(reader.read().ok_or(TTFontLoadError::MalformedHeadTable)?);
        Ok(())
    }

    pub(crate) fn load_loca(&mut self) -> Result<(), TTFontLoadError> {
        let loca_data = self
            .get_table_data(b"loca")
            .ok_or(TTFontLoadError::MissingRequiredTable("loca".into()))?;
        let mut reader = ByteReader::new(&self.data).subreader(loca_data);
        self.loca_table = Some(
            LocaTable::load(&mut reader, self.maxp_table(), self.head_table())
                .ok_or(TTFontLoadError::MalformedLoca)?,
        );
        Ok(())
    }

    pub(crate) fn get_table_directory(
        &self,
        name: &(impl Into<TableHeader> + Copy),
    ) -> Option<TableDirectory> {
        self.table_directories.get(&(*name).into()).copied()
    }

    pub(crate) fn get_table_data(
        &self,
        name: &(impl Into<TableHeader> + Copy),
    ) -> Option<BytesRef> {
        let table_directory = self.get_table_directory(name)?;
        let offset = table_directory.offset as usize;
        let length = table_directory.length as usize;
        assert!(self.data.get(offset..(offset + length)).is_some());
        Some(BytesRef { offset, length })
    }

    pub(crate) fn cmap_table(&self) -> &CmapTable {
        self.cmap_table.as_ref().unwrap()
    }

    pub(crate) fn head_table(&self) -> &HeadTable {
        self.head_table.as_ref().unwrap()
    }

    pub(crate) fn maxp_table(&self) -> &MaxpTable {
        self.maxp_table.as_ref().unwrap()
    }

    pub(crate) fn loca_table(&self) -> &LocaTable {
        self.loca_table.as_ref().unwrap()
    }

    pub(crate) fn search_glyph_index(&self, codepoint: u16) -> u16 {
        let reader = ByteReader::new(&self.data);
        let cmap = self.cmap_table();
        for table in &cmap.format4_tables {
            let glyph_index = table.search_for_codepoint(codepoint, &reader);
            if glyph_index != 0 {
                return glyph_index;
            }
        }
        0
    }

    /// Offset for glyph.
    pub(crate) fn glyph_location(&self, codepoint: u32) -> Option<u32> {
        let glyph_index = self.search_glyph_index(codepoint as u16);
        let loca_data = self.get_table_data(b"loca").unwrap();
        let loca_table = self.loca_table();
        let reader: &mut ByteReader = &mut ByteReader::new(&self.data).subreader(loca_data);
        let size = match loca_table.is_long {
            true => 4,
            false => 2,
        };
        reader.goto(glyph_index as usize * size);
        match loca_table.is_long {
            true => reader.read::<u32>(),
            false => reader.read::<u16>().map(u32::from),
        }
    }

    pub(crate) fn get_glyph(&self, codepoint: u32) -> Option<SimpleGlyph> {
        let location = self.glyph_location(codepoint)?;
        let root_reader = ByteReader::new(&self.data);
        let glyf_table = self.get_table_data(b"glyf")?;
        let mut glyf_data_reader = root_reader.subreader(glyf_table);
        glyf_data_reader.goto(location as usize);
        SimpleGlyph::read_from(&mut glyf_data_reader)
    }
}
