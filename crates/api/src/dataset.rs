// Memory-mapped reference dataset.
//
//     ┌──────────── header (32 B) ─────────────┐
//     │ magic[4]="R26V"  version(u32)=1       │
//     │ count(u32)       dims(u32)=14         │
//     │ stride(u32)=16   dtype(u32) (1=f16)   │
//     │ reserved[8]                           │
//     ├───────────────────────────────────────┤
//     │ vectors:  count × stride × sizeof(d)  │
//     ├───────────────────────────────────────┤
//     │ labels:   ceil(count / 8) bytes       │
//     └───────────────────────────────────────┘
//
// The api crate only ever reads f16-encoded datasets in production; the f32
// path is kept around for debugging and as a fallback if f16 introduces
// detection regressions.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"R26V";
pub const VERSION: u32 = 1;
pub const DIMS: usize = 14;
pub const STRIDE: usize = 16;
pub const DTYPE_F16: u32 = 1;
pub const DTYPE_F32: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub count: u32,
    pub dims: u32,
    pub stride: u32,
    pub dtype: u32,
    pub reserved: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<Header>() == 32);

pub enum VectorStore {
    /// raw f16 bits, count × STRIDE u16
    F16(&'static [u16]),
    /// f32, count × STRIDE
    F32(&'static [f32]),
}

pub struct Dataset {
    pub header: Header,
    pub vectors: VectorStore,
    pub labels: &'static [u8],
}

struct Mmap {
    ptr: *const u8,
    len: usize,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Mmap {
    fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty dataset"));
        }
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        unsafe { libc::madvise(ptr, len, libc::MADV_WILLNEED); }
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    #[inline(always)]
    fn as_static(&self) -> &'static [u8] {
        // SAFETY: we leak the Mmap; lives for the program lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Dataset {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mmap = Mmap::open(path)?;
        let bytes: &'static [u8] = Box::leak(Box::new(mmap)).as_static();

        if bytes.len() < std::mem::size_of::<Header>() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "dataset too small"));
        }
        let header: Header = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Header) };
        if &header.magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
        }
        if header.version != VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad version"));
        }
        if header.dims as usize != DIMS {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected dims"));
        }
        if (header.stride as usize) < DIMS {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "stride < dims"));
        }
        let stride = header.stride as usize;
        let count = header.count as usize;
        let bytes_per_lane = match header.dtype {
            DTYPE_F16 => 2,
            DTYPE_F32 => 4,
            _ => return Err(io::Error::new(io::ErrorKind::Unsupported, "unsupported dtype")),
        };
        let vectors_bytes = count
            .checked_mul(stride)
            .and_then(|n| n.checked_mul(bytes_per_lane))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "size overflow"))?;
        let label_bytes = (count + 7) / 8;
        let needed = 32 + vectors_bytes + label_bytes;
        if bytes.len() < needed {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated dataset"));
        }

        let vectors_slice = &bytes[32..32 + vectors_bytes];
        let vectors = match header.dtype {
            DTYPE_F16 => {
                let ptr = vectors_slice.as_ptr() as *const u16;
                VectorStore::F16(unsafe { std::slice::from_raw_parts(ptr, count * stride) })
            }
            DTYPE_F32 => {
                let ptr = vectors_slice.as_ptr() as *const f32;
                VectorStore::F32(unsafe { std::slice::from_raw_parts(ptr, count * stride) })
            }
            _ => unreachable!(),
        };
        let labels: &'static [u8] = &bytes[32 + vectors_bytes..32 + vectors_bytes + label_bytes];

        Ok(Self { header, vectors, labels })
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.header.count as usize
    }

    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.header.stride as usize
    }

    #[inline(always)]
    pub fn is_fraud(&self, idx: usize) -> bool {
        let byte = unsafe { *self.labels.get_unchecked(idx >> 3) };
        ((byte >> (idx & 7)) & 1) != 0
    }

    /// Touch every page so subsequent queries don't pay page-fault latency.
    pub fn warm_up(&self) -> u64 {
        let mut s: u64 = 0;
        match &self.vectors {
            VectorStore::F16(v) => {
                for &x in v.iter().step_by(1024) { s = s.wrapping_add(x as u64); }
            }
            VectorStore::F32(v) => {
                for &x in v.iter().step_by(1024) { s = s.wrapping_add(x.to_bits() as u64); }
            }
        }
        for &x in self.labels.iter().step_by(64) { s = s.wrapping_add(x as u64); }
        s
    }
}
