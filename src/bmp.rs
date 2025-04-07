#![allow(dead_code)]

use std::{
    fs::File,
    io::{self, BufWriter, Write as _},
    path::Path,
};

use cgmath::Vector2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Rgba8,
    Rgba8Srgb,
    Bgra8,
    Bgra8Srgb,
}

fn gamma_correct(u_in: u8, gamma: f32) -> u8 {
    let f_in = (u_in as f32) / 255.0;
    let f_out = f_in.powf(gamma);
    (f_out * 255.0).floor() as u8
}

fn srgb_to_rgb(srgb: u8) -> u8 {
    gamma_correct(srgb, 1.0 / 2.2)
}

fn bmp_header(data: &mut Vec<u8>, size: Vector2<u32>) {
    data.extend_from_slice(b"BM"); /* Signature */
    data.extend_from_slice(&(size.x * 4 * size.y + 54u32).to_le_bytes()); /* File size */
    data.extend_from_slice(&0u16.to_le_bytes()); /* Reserved */
    data.extend_from_slice(&0u16.to_le_bytes()); /* Reserved */
    data.extend_from_slice(&54u32.to_le_bytes()); /* Data offset */
    data.extend_from_slice(&40u32.to_le_bytes()); /* Size of InfoHeader (=40) */
    data.extend_from_slice(&size.x.to_le_bytes()); /* Width */
    data.extend_from_slice(&size.y.to_le_bytes()); /* Height */
    data.extend_from_slice(&1u16.to_le_bytes()); /* Planes (=1) */
    data.extend_from_slice(&32u16.to_le_bytes()); /* Bits Per Pixel */
    data.extend_from_slice(&0u32.to_le_bytes()); /* Compression */
    data.extend_from_slice(&0u32.to_le_bytes()); /* Image Size (allowed as 0 if compression is 0) */
    data.extend_from_slice(&2835u32.to_le_bytes()); /* X pixels per meter (~=72ppi) */
    data.extend_from_slice(&2835u32.to_le_bytes()); /* Y pixels per meter (~=72ppi) */
    data.extend_from_slice(&0u32.to_le_bytes()); /* Colors used */
    data.extend_from_slice(&0u32.to_le_bytes()); /* Important colors */
}

/// Encode BMP with a a custom pixel format.
pub fn encode_bmp_with<T>(
    size: Vector2<u32>,
    mut pixels: impl Iterator<Item = T>,
    mut f_encode: impl FnMut(T) -> [u8; 4],
) -> Vec<u8> {
    let mut data = Vec::new();
    bmp_header(&mut data, size);
    let width = size.x as usize;
    let height = size.y as usize;
    data.resize(width * height * 4 + 54, 0);
    for y in (0..height).rev() {
        for x in 0..width {
            let rgba = pixels.next().unwrap();
            let encoded = f_encode(rgba);
            let i_min = y * width * 4 + x * 4 + 54;
            data[i_min] = encoded[0];
            data[i_min + 1] = encoded[1];
            data[i_min + 2] = encoded[2];
            data[i_min + 3] = encoded[3];
        }
    }
    data
}

/// Encode an image data into BMP format.
pub fn encode_bmp(size: Vector2<u32>, format: PixelFormat, pixel_data: &[u8]) -> Vec<u8> {
    match format {
        PixelFormat::Rgba8 => {
            encode_bmp_with(size, pixel_data.array_chunks::<4>().copied(), |src| {
                [src[2], src[1], src[0], src[3]]
            })
        }
        PixelFormat::Rgba8Srgb => {
            encode_bmp_with(size, pixel_data.array_chunks::<4>().copied(), |src| {
                [src[2], src[1], src[0], src[3]].map(srgb_to_rgb)
            })
        }
        PixelFormat::Bgra8 => {
            encode_bmp_with(size, pixel_data.array_chunks::<4>().copied(), |src| {
                [src[0], src[1], src[2], src[3]]
            })
        }
        PixelFormat::Bgra8Srgb => {
            encode_bmp_with(size, pixel_data.array_chunks::<4>().copied(), |src| {
                [src[0], src[1], src[2], src[3]].map(srgb_to_rgb)
            })
        }
    }
}

fn save_data(path: impl AsRef<Path>, data: &[u8]) -> io::Result<()> {
    let file = File::options()
        .read(false)
        .write(true)
        .append(false)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(data)
}

/// Save an image into a `.bmp` file for debug.
pub fn save_bmp(
    path: impl AsRef<Path>,
    size: Vector2<u32>,
    format: PixelFormat,
    pixel_data: &[u8],
) -> io::Result<()> {
    save_data(path, &encode_bmp(size, format, pixel_data))
}

/// Save an image into a `.bmp` file for debug.
pub fn save_bmp_with<T>(
    path: impl AsRef<Path>,
    size: Vector2<u32>,
    pixels: impl Iterator<Item = T>,
    f_encode: impl FnMut(T) -> [u8; 4],
) -> io::Result<()> {
    save_data(path, &encode_bmp_with(size, pixels, f_encode))
}
