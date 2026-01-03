//! Length-Prefixed Frame Encoding
//!
//! Provides reliable message boundaries over stream-based IPC (stdin/stdout).

use rkyv::ser::serializers::AllocSerializer;
use rkyv::validation::validators::DefaultValidator;
use rkyv::{Archive, CheckBytes, Deserialize, Infallible, Serialize};
use std::io::{BufReader, BufWriter, Read, Write};
use thiserror::Error;

/// Maximum frame size (16 MB) to prevent memory exhaustion
pub const MAX_FRAME_SIZE: usize = 16 * 1024 * 1024;

/// Errors that can occur during frame encoding/decoding
#[derive(Debug, Error)]
pub enum FrameError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Frame too large: {size} bytes (max {max} bytes)")]
    FrameTooLarge { size: usize, max: usize },

    #[error("Invalid frame: {0}")]
    InvalidFrame(String),

    #[error("End of stream")]
    EndOfStream,
}

/// Write a message with length prefix to a writer
///
/// Frame format:
/// ```text
/// +----------------+------------------+
/// | length (4 LE)  | rkyv payload     |
/// +----------------+------------------+
/// ```
pub fn write_frame<W, T>(writer: &mut BufWriter<W>, message: &T) -> Result<(), FrameError>
where
    W: Write,
    T: Serialize<AllocSerializer<256>>,
{
    // Serialize the message
    let bytes = rkyv::to_bytes::<_, 256>(message)
        .map_err(|e| FrameError::Serialization(e.to_string()))?;

    let len = bytes.len();
    if len > MAX_FRAME_SIZE {
        return Err(FrameError::FrameTooLarge {
            size: len,
            max: MAX_FRAME_SIZE,
        });
    }

    // Write length prefix (4 bytes, little-endian)
    writer.write_all(&(len as u32).to_le_bytes())?;

    // Write payload
    writer.write_all(&bytes)?;

    // Flush to ensure message is sent
    writer.flush()?;

    Ok(())
}

/// Read a message with length prefix from a reader
pub fn read_frame<R, T>(reader: &mut BufReader<R>) -> Result<T, FrameError>
where
    R: Read,
    T: Archive,
    T::Archived: for<'a> CheckBytes<DefaultValidator<'a>> + Deserialize<T, Infallible>,
{
    // Read length prefix
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            return Err(FrameError::EndOfStream);
        }
        Err(e) => return Err(FrameError::Io(e)),
    }

    let len = u32::from_le_bytes(len_buf) as usize;

    // Validate length
    if len > MAX_FRAME_SIZE {
        return Err(FrameError::FrameTooLarge {
            size: len,
            max: MAX_FRAME_SIZE,
        });
    }

    if len == 0 {
        return Err(FrameError::InvalidFrame("zero-length frame".to_string()));
    }

    // Read payload into aligned buffer
    let mut buf = rkyv::AlignedVec::with_capacity(len);
    buf.resize(len, 0);
    reader.read_exact(&mut buf)?;

    // Validate and access archived data
    let archived = rkyv::check_archived_root::<T>(&buf)
        .map_err(|e| FrameError::Deserialization(e.to_string()))?;

    // Deserialize
    let value: T = archived
        .deserialize(&mut Infallible)
        .expect("infallible deserialization");

    Ok(value)
}

/// Frame writer wrapper for convenient message sending
pub struct FrameWriter<W: Write> {
    writer: BufWriter<W>,
}

impl<W: Write> FrameWriter<W> {
    /// Create a new frame writer
    pub fn new(writer: W) -> Self {
        Self {
            writer: BufWriter::with_capacity(64 * 1024, writer), // 64KB buffer
        }
    }

    /// Write a message
    pub fn write<T>(&mut self, message: &T) -> Result<(), FrameError>
    where
        T: Serialize<AllocSerializer<256>>,
    {
        write_frame(&mut self.writer, message)
    }

    /// Flush the underlying writer
    pub fn flush(&mut self) -> Result<(), FrameError> {
        self.writer.flush()?;
        Ok(())
    }

    /// Get mutable reference to the inner writer
    pub fn inner_mut(&mut self) -> &mut BufWriter<W> {
        &mut self.writer
    }

    /// Consume and return the inner writer
    pub fn into_inner(self) -> BufWriter<W> {
        self.writer
    }
}

/// Frame reader wrapper for convenient message receiving
pub struct FrameReader<R: Read> {
    reader: BufReader<R>,
}

impl<R: Read> FrameReader<R> {
    /// Create a new frame reader
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::with_capacity(64 * 1024, reader), // 64KB buffer
        }
    }

    /// Read a message
    pub fn read<T>(&mut self) -> Result<T, FrameError>
    where
        T: Archive,
        T::Archived: for<'a> CheckBytes<DefaultValidator<'a>> + Deserialize<T, Infallible>,
    {
        read_frame(&mut self.reader)
    }

    /// Check if the buffer has any data available
    pub fn has_buffered_data(&self) -> bool {
        !self.reader.buffer().is_empty()
    }

    /// Get mutable reference to the inner reader
    pub fn inner_mut(&mut self) -> &mut BufReader<R> {
        &mut self.reader
    }

    /// Consume and return the inner reader
    pub fn into_inner(self) -> BufReader<R> {
        self.reader
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
    use std::io::Cursor;

    #[derive(Debug, Clone, PartialEq, Archive, RkyvSerialize, RkyvDeserialize)]
    #[archive(check_bytes)]
    struct TestMessage {
        value: u64,
        text: String,
    }

    #[test]
    fn test_roundtrip() {
        let original = TestMessage {
            value: 42,
            text: "hello world".to_string(),
        };

        // Write to buffer
        let mut buffer = Vec::new();
        {
            let mut writer = FrameWriter::new(&mut buffer);
            writer.write(&original).unwrap();
        }

        // Read back
        let mut reader = FrameReader::new(Cursor::new(buffer));
        let decoded: TestMessage = reader.read().unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_multiple_messages() {
        let messages = vec![
            TestMessage {
                value: 1,
                text: "first".to_string(),
            },
            TestMessage {
                value: 2,
                text: "second".to_string(),
            },
            TestMessage {
                value: 3,
                text: "third".to_string(),
            },
        ];

        // Write all messages
        let mut buffer = Vec::new();
        {
            let mut writer = FrameWriter::new(&mut buffer);
            for msg in &messages {
                writer.write(msg).unwrap();
            }
        }

        // Read all back
        let mut reader = FrameReader::new(Cursor::new(buffer));
        for expected in &messages {
            let decoded: TestMessage = reader.read().unwrap();
            assert_eq!(expected, &decoded);
        }
    }

    #[test]
    fn test_end_of_stream() {
        let buffer: Vec<u8> = Vec::new();
        let mut reader = FrameReader::new(Cursor::new(buffer));
        let result: Result<TestMessage, _> = reader.read();
        assert!(matches!(result, Err(FrameError::EndOfStream)));
    }
}
