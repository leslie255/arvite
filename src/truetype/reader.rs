use font_types::{F2Dot14, FWord, Fixed, LongDateTime, UfWord};

use crate::generator;

/// I'm not a big fan of this design. Might change later.
/// This seemed like an OK thing at the time of designing.
#[derive(Debug, Clone, Default)]
pub(crate) struct ByteReader<'a> {
    /// If this `ByteReader` is a sub reader, this stores its offset relative to the root reader.
    /// This is useful for indexing with `BytesRef`.
    relative_to_root: Option<usize>,
    bytes: &'a [u8],
    /// Cursor is one-past.
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            relative_to_root: None,
            bytes,
            cursor: 0,
        }
    }

    pub(crate) fn is_end(&self) -> bool {
        self.cursor >= self.bytes.len()
    }

    pub(crate) fn goto(&mut self, cursor: usize) {
        self.cursor = cursor.min(self.bytes.len());
    }

    pub(crate) fn read_byte(&mut self) -> Option<u8> {
        if let Some(&byte) = self.bytes.get(self.cursor) {
            self.cursor += 1;
            Some(byte)
        } else {
            None
        }
    }

    /// Returns `None` if remaining bytes is less than `N`, returns `None`.
    pub(crate) fn read_array<const N: usize>(&mut self) -> Option<[u8; N]> {
        if let Some(bytes) = self.bytes.get(self.cursor..self.cursor + N) {
            self.cursor += N;
            let mut array = [0u8; N];
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), array.as_mut_ptr(), N);
            }
            Some(array)
        } else {
            None
        }
    }

    pub(crate) fn read_bytes(&mut self, n_bytes: usize) -> Option<BytesRef> {
        if self.bytes.get(self.cursor..self.cursor + n_bytes).is_some() {
            let bytes_ref = BytesRef {
                offset: self.cursor + self.relative_to_root.unwrap_or(0),
                length: n_bytes,
            };
            self.cursor += n_bytes;
            Some(bytes_ref)
        } else {
            None
        }
    }


    #[track_caller]
    pub(crate) fn subreader(&self, bytes_ref: BytesRef) -> ByteReader<'a> {
        ByteReader {
            relative_to_root: Some(self.relative_to_root.unwrap_or(bytes_ref.offset)),
            bytes: {
                let bytes_ref = BytesRef {
                    offset: bytes_ref.offset - self.relative_to_root.unwrap_or(0),
                    ..bytes_ref
                };
                bytes_ref.get(self.bytes)
            },
            cursor: 0,
        }
    }

    pub(crate) fn peek_byte(&mut self) -> Option<u8> {
        let mut self_ = self.clone();
        self_.read_byte()
    }

    pub(crate) fn peek_bytes(&mut self, n: usize) -> Option<BytesRef> {
        let mut self_ = self.clone();
        self_.read_bytes(n)
    }

    pub(crate) fn peek_array<const N: usize>(&mut self) -> Option<[u8; N]> {
        let mut self_ = self.clone();
        self_.read_array()
    }

    pub(crate) fn peek<T: ReadFrom>(&mut self) -> Option<T> {
        let mut self_ = self.clone();
        self_.read()
    }

    pub(crate) fn read<T: ReadFrom>(&mut self) -> Option<T> {
        T::read_from(self)
    }

    pub(crate) fn read_multiple<T: ReadFrom>(&mut self, n: usize) -> impl Iterator<Item = T> {
        generator! {
            for _ in 0..n {
                if let Some(x) = self.read::<T>() {
                    yield x;
                } else {
                    break;
                }
            }
        }
    }
}

pub(crate) trait ReadFrom: Sized {
    fn read_from(reader: &mut ByteReader) -> Option<Self>;
}

macro_rules! impl_read_from_for_int {
    ($T:ty, $read_T_be:ident $(,)?) => {
        impl ReadFrom for $T {
            fn read_from(reader: &mut ByteReader) -> Option<Self> {
                reader.read_array().map(Self::from_be_bytes)
            }
        }
    };
}

impl_read_from_for_int!(u128, read_u128_be);
impl_read_from_for_int!(u64, read_u64_be);
impl_read_from_for_int!(u32, read_u32_be);
impl_read_from_for_int!(u16, read_u16_be);
impl_read_from_for_int!(u8, read_u8_be);
impl_read_from_for_int!(i128, read_i128_be);
impl_read_from_for_int!(i64, read_i64_be);
impl_read_from_for_int!(i32, read_i32_be);
impl_read_from_for_int!(i16, read_i16_be);
impl_read_from_for_int!(i8, read_i8_be);

#[macro_export]
macro_rules! impl_read_from {
    ($T:ty, $($fields:ident),* $(,)?) => {
        impl ReadFrom for $T {
            fn read_from(reader: &mut ByteReader) -> Option<Self> {
                Some(Self {
                    $($fields: ReadFrom::read_from(reader)?),*
                })
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BytesRef {
    pub(crate) offset: usize,
    pub(crate) length: usize,
}

impl BytesRef {
    #[track_caller]
    pub(crate) fn get(self, bytes: &[u8]) -> &[u8] {
        &bytes[self.offset..self.offset + self.length]
    }
}

/// `BytesRef` with a static length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixedBytesRef<const N: usize> {
    pub(crate) offset: usize,
}

impl<const N: usize> FixedBytesRef<N> {
    pub(crate) fn get(self, bytes: &[u8]) -> &[u8] {
        &bytes[self.offset..self.offset + N]
    }
}

impl<const N: usize> ReadFrom for FixedBytesRef<N> {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        let bytes_ref = reader.read_bytes(N)?;
        debug_assert!(bytes_ref.length == N);
        Some(Self {
            offset: bytes_ref.offset,
        })
    }
}

impl ReadFrom for FWord {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self::new)
    }
}

impl ReadFrom for UfWord {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self::new)
    }
}

impl ReadFrom for F2Dot14 {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self::from_bits)
    }
}

impl ReadFrom for Fixed {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self::from_bits)
    }
}

impl ReadFrom for LongDateTime {
    fn read_from(reader: &mut ByteReader) -> Option<Self> {
        reader.read().map(Self::new)
    }
}
