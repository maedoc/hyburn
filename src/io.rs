//! IO layer: NPY tensor bridge and file utilities.
//!
//! Implements a minimal NPY reader/writer to avoid ndarray version conflicts
//! with burn's internal ndarray dependency. The format is simple:
//! magic + header + raw f32 data.
//!
//! File I/O functions (read_npy_f32, write_npy_f32) are not available in WASM builds.
//! Pure tensor conversion functions (ndarray_to_tensor, tensor_to_flat_f32) work everywhere.

use burn::prelude::Backend;
use burn::tensor::{Tensor, TensorData};
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Write};
use std::path::Path;
use crate::error::{Result, SimulationError};

/// NPY magic prefix
const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Read a `.npy` file into a `Vec<f32>` + shape.
///
/// Supports little-endian float32 arrays (the common case from NumPy).
/// Not available in WASM builds (no filesystem access).
#[cfg(not(target_arch = "wasm32"))]
pub fn read_npy_f32<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, Vec<usize>)> {
    let mut buf = Vec::new();
    let mut f = std::fs::File::open(path.as_ref())?;
    f.read_to_end(&mut buf)?;

    // Check magic
    if &buf[0..6] != NPY_MAGIC {
        return Err(SimulationError::InvalidState("Not a valid .npy file".into()));
    }

    // Parse version byte
    let _major = buf[6];
    let _minor = buf[7];

    // Header length: 2 bytes LE (for version 1.x) or 4 bytes LE (version 2.x/3.x)
    let header_len = if _major < 2 {
        let len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        10 + len
    } else {
        let len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
        12 + len
    };

    // Parse header string to extract shape and dtype
    let header_str = String::from_utf8_lossy(&buf[10..header_len.min(buf.len())]);
    let shape = parse_npy_shape(&header_str)?;
    let descr = parse_npy_descr(&header_str)?;

    if descr != "<f4" && descr != "|f4" && descr != "f4" && descr != "<f8" {
        return Err(SimulationError::InvalidState(format!(
            "Unsupported NPY dtype descriptor: '{}', only f32/f64 supported", descr
        )));
    }

    let data_start = header_len;
    let total_elements: usize = shape.iter().product();

    let data = if descr == "<f8" || descr == "|f8" || descr == "f8" {
        // f64 → f32 conversion
        let bytes = &buf[data_start..];
        if bytes.len() < total_elements * 8 {
            return Err(SimulationError::InvalidState("NPY data truncated (f64)".into()));
        }
        bytes.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()
    } else {
        // f32
        let bytes = &buf[data_start..];
        if bytes.len() < total_elements * 4 {
            return Err(SimulationError::InvalidState("NPY data truncated (f32)".into()));
        }
        bytes.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };

    Ok((data, shape))
}

/// Write a flat f32 array + shape to a `.npy` file.
/// Not available in WASM builds (no filesystem access).
#[cfg(not(target_arch = "wasm32"))]
pub fn write_npy_f32<P: AsRef<Path>>(path: P, data: &[f32], shape: &[usize]) -> Result<()> {
    let mut f = std::fs::File::create(path.as_ref())?;

    // Build header string
    let shape_str = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {} }}",
        shape_str
    );

    // Pad header to be divisible by 64 (for alignment), including 10-byte prefix and newline
    let mut header = header;
    let total_with_newline = header.len() + 10 + 1;
    let remainder = total_with_newline % 64;
    let padding_needed = if remainder == 0 { 0 } else { 64 - remainder };
    for _ in 0..padding_needed {
        header.push(' ');
    }
    header.push('\n');

    // Write magic + version + header length + header
    f.write_all(NPY_MAGIC)?;
    f.write_all(&[1u8, 0])?; // version 1.0
    let hlen = header.len() as u16;
    f.write_all(&hlen.to_le_bytes())?;
    f.write_all(header.as_bytes())?;

    // Write data
    for &val in data {
        f.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

/// Convert data read by `read_npy_f32` into a Burn tensor.
pub fn ndarray_to_tensor<B: Backend, const N: usize>(
    data: Vec<f32>,
    shape: Vec<usize>,
    device: &B::Device,
) -> Tensor<B, N> {
    assert!(shape.len() == N, "Expected {N}D array, got {}D", shape.len());
    Tensor::from_floats(
        TensorData::new::<f32, Vec<usize>>(data, shape),
        device,
    )
}

/// Convert a Burn `Tensor<B, N>` to flat f32 data + shape.
pub fn tensor_to_flat_f32<B: Backend, const N: usize>(
    tensor: Tensor<B, N>,
) -> (Vec<f32>, Vec<usize>) {
    let data = tensor.into_data();
    let shape: Vec<usize> = data.shape.to_vec();
    let values: Vec<f32> = data.as_slice::<f32>().unwrap().to_vec();
    (values, shape)
}

/// Extract a Python-quoted key's value from an NPY header using a state machine.
/// Only matches key positions that are outside of any string literal, preventing
/// false matches when `'key':` appears as part of a string value.
/// Returns `None` if the key is not found, or if the value is malformed.
fn find_quoted_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let bytes = header.as_bytes();
    let key_bytes = key.as_bytes();
    let n = bytes.len();
    let mut in_string = false;
    let mut i = 0;

    while i < n {
        let ch = bytes[i];

        // Check for quoted key FIRST (before toggling in_string)
        if !in_string && (ch == b'\'' || ch == b'\"') {
            let quote = ch;
            if i + 1 + key_bytes.len() + 1 < n 
                && bytes[i+1..].starts_with(key_bytes) 
                && bytes[i + 1 + key_bytes.len()] == quote
                && bytes[i + 1 + key_bytes.len() + 1] == b':' 
            {
                // Found quoted key 'key':, extract value
                let val_start = i + 1 + key_bytes.len() + 2; // skip 'key':
                let after_colon = &bytes[val_start..];
                let after_colon_str = std::str::from_utf8(after_colon).ok()?;
                let trimmed = after_colon_str.trim_start();

                if trimmed.starts_with('(') {
                    let mut depth = 0;
                    let mut j = 0;
                    for (k, c) in trimmed.char_indices() {
                        if c == '(' { depth += 1; }
                        if c == ')' { depth -= 1; }
                        if depth == 0 && c == ')' {
                            j = k;
                            break;
                        }
                    }
                    return Some(trimmed[1..j].trim());
                } else if trimmed.starts_with('\'') || trimmed.starts_with('"') {
                    let quote = trimmed.chars().next()?;
                    let after_quote = &trimmed[1..];
                    if let Some(end_pos) = after_quote.find(quote) {
                        return Some(&after_quote[..end_pos]);
                    }
                }
            }
        }

        // Toggle in_string for regular string content
        if ch == b'\'' || ch == b'\"' {
            in_string = !in_string;
        }

        i += 1;
    }
    None
}

/// Parse shape from NPY header string like "{'descr': '<f4', ... 'shape': (2, 3) }"
fn parse_npy_shape(header: &str) -> Result<Vec<usize>> {
    let inner = find_quoted_value(header, "shape")
        .ok_or_else(|| SimulationError::InvalidState("No 'shape' key in NPY header".into()))?;
    let shape: Vec<usize> = inner.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| SimulationError::InvalidState(format!("Failed to parse NPY shape: {}", e)))?;
    Ok(shape)
}

/// Parse dtype descriptor from NPY header string
fn parse_npy_descr(header: &str) -> Result<String> {
    let value = find_quoted_value(header, "descr")
        .ok_or_else(|| SimulationError::InvalidState("No 'descr' key in NPY header".into()))?;
    Ok(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;
    use tempfile::tempdir;

    type B = NdArray<f32>;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_npy_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.npy");
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        write_npy_f32(&path, &data, &shape).unwrap();
        let (data2, shape2) = read_npy_f32(&path).unwrap();
        assert_eq!(data, data2);
        assert_eq!(shape, shape2);
    }

    #[test]
    fn test_ndarray_to_tensor() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        let tensor = ndarray_to_tensor::<B, 2>(data.clone(), shape.clone(), &Default::default());
        let (data2, shape2) = tensor_to_flat_f32::<B, 2>(tensor);
        assert_eq!(data, data2);
        assert_eq!(shape, shape2);
    }
}
