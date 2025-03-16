use font_types::Fixed;

use super::*;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MaxpTable {
    /// 0x00010000 (1.0)
    pub(crate) version: Fixed,
    /// the number of glyphs in the font
    pub(crate) num_glyphs: u16,
    /// points in non-compound glyph
    pub(crate) max_points: u16,
    /// contours in non-compound glyph
    pub(crate) max_contours: u16,
    /// points in compound glyph
    pub(crate) max_component_points: u16,
    /// contours in compound glyph
    pub(crate) max_component_contours: u16,
    /// set to 2
    pub(crate) max_zones: u16,
    /// points used in Twilight Zone (Z0)
    pub(crate) max_twilight_points: u16,
    /// number of Storage Area locations
    pub(crate) max_storage: u16,
    /// number of FDEFs
    pub(crate) max_function_defs: u16,
    /// number of IDEFs
    pub(crate) max_instruction_defs: u16,
    /// maximum stack depth
    pub(crate) max_stack_elements: u16,
    /// byte count for glyph instructions
    pub(crate) max_size_of_instructions: u16,
    /// number of glyphs referenced at top level
    pub(crate) max_component_elements: u16,
    /// levels of recursion, set to 0 if font has only simple glyphs
    pub(crate) max_component_depth: u16,
}

impl_read_from!(
    MaxpTable,
    version,
    num_glyphs,
    max_points,
    max_contours,
    max_component_points,
    max_component_contours,
    max_zones,
    max_twilight_points,
    max_storage,
    max_function_defs,
    max_instruction_defs,
    max_stack_elements,
    max_size_of_instructions,
    max_component_elements,
    max_component_depth,
);
