// Memory-mapped reference dataset.
//
//     ┌──────────── header (32 B) ─────────────┐
//     │ magic[4]="R26V"  version(u32)=2       │
//     │ count(u32)       dims(u32)=14         │
//     │ stride(u32)=16   dtype(u32) (1=f16)   │
//     │ nlist(u32)       flags(u32)           │
//     ├───────────────────────────────────────┤
//     │ vectors:  count × stride × sizeof(d)  │  ← reordered by cluster (IVF)
//     ├───────────────────────────────────────┤
//     │ labels:   ceil(count / 8) bytes       │  ← reordered to match
//     ├───────────────────────────────────────┤
//     │ centroids (if FLAG_IVF):              │
//     │     nlist × stride × sizeof(d)        │
//     ├───────────────────────────────────────┤
//     │ offsets   (if FLAG_IVF):              │
//     │     (nlist + 1) × u32                 │
//     └───────────────────────────────────────┘

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub const MAGIC: &[u8; 4] = b"R26V";
pub const VERSION_V1: u32 = 1;
pub const VERSION_V2: u32 = 2;
pub const DIMS: usize = 14;
pub const STRIDE: usize = 16;
pub const DTYPE_F16: u32 = 1;
pub const DTYPE_F32: u32 = 2;
pub const FLAG_IVF: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    pub magic: [u8; 4],
    pub version: u32,
    pub count: u32,
    pub dims: u32,
    pub stride: u32,
    pub dtype: u32,
    pub nlist: u32,
    pub flags: u32,
}

const _: () = assert!(std::mem::size_of::<Header>() == 32);

pub enum VectorStore {
    F16(&'static [u16]),
    F32(&'static [f32]),
}

pub struct Dataset {
    pub header: Header,
    pub vectors: VectorStore,
    pub labels: &'static [u8],
    pub ivf: Option<IvfIndex>,
}

/// Centroids decoded into f32 (small — at most 1024 × 16 × 4 = 64 KB) so the
/// stage-1 hot loop doesn't pay for f16→f32 conversions on every query.
pub struct IvfIndex {
    pub nlist: usize,
    pub centroids_f32: Vec<f32>,
    pub offsets: &'static [u32],
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
        unsafe {
            // Try MADV_POPULATE_READ (Linux 5.14+) first — synchronous
            // page-in. Falls back to MADV_WILLNEED (async hint) if the
            // kernel doesn't know the flag. Avoids page-fault latency
            // on the first query against any cold page.
            //
            // libc::MADV_POPULATE_READ may not be exposed depending on the
            // version we picked up; use the well-known constant directly.
            const MADV_POPULATE_READ: i32 = 22;
            if libc::madvise(ptr, len, MADV_POPULATE_READ) != 0 {
                libc::madvise(ptr, len, libc::MADV_WILLNEED);
            }
        }
        Ok(Mmap { ptr: ptr as *const u8, len })
    }

    #[inline(always)]
    fn as_static(&self) -> &'static [u8] {
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
        if header.version != VERSION_V1 && header.version != VERSION_V2 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "unsupported version"));
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

        let ivf_present = header.version >= VERSION_V2 && (header.flags & FLAG_IVF) != 0;
        let nlist = header.nlist as usize;
        let centroids_bytes = if ivf_present { nlist * stride * bytes_per_lane } else { 0 };
        let offsets_bytes = if ivf_present { (nlist + 1) * 4 } else { 0 };

        let needed = 32 + vectors_bytes + label_bytes + centroids_bytes + offsets_bytes;
        if bytes.len() < needed {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "truncated dataset"));
        }

        let mut cursor = 32;
        let vectors_slice = &bytes[cursor..cursor + vectors_bytes];
        cursor += vectors_bytes;
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

        let labels: &'static [u8] = &bytes[cursor..cursor + label_bytes];
        cursor += label_bytes;

        let ivf = if ivf_present {
            let centroids_slice = &bytes[cursor..cursor + centroids_bytes];
            cursor += centroids_bytes;
            let offsets_slice = &bytes[cursor..cursor + offsets_bytes];
            // cursor += offsets_bytes;

            let mut centroids_f32 = vec![0f32; nlist * stride];
            match header.dtype {
                DTYPE_F16 => {
                    let cs = unsafe {
                        std::slice::from_raw_parts(
                            centroids_slice.as_ptr() as *const u16,
                            nlist * stride,
                        )
                    };
                    for (dst, &src) in centroids_f32.iter_mut().zip(cs) {
                        *dst = half::f16::from_bits(src).to_f32();
                    }
                }
                DTYPE_F32 => {
                    let cs = unsafe {
                        std::slice::from_raw_parts(
                            centroids_slice.as_ptr() as *const f32,
                            nlist * stride,
                        )
                    };
                    centroids_f32.copy_from_slice(cs);
                }
                _ => unreachable!(),
            }

            let offsets: &'static [u32] = unsafe {
                std::slice::from_raw_parts(offsets_slice.as_ptr() as *const u32, nlist + 1)
            };

            Some(IvfIndex { nlist, centroids_f32, offsets })
        } else {
            None
        };

        Ok(Self { header, vectors, labels, ivf })
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

    /// Touch every 4 KB page so subsequent queries don't pay page-fault
    /// latency. Belt-and-suspenders alongside MADV_POPULATE_READ in `Mmap::open`.
    pub fn warm_up(&self) -> u64 {
        let mut s: u64 = 0;
        // 1 u16 = 2 bytes → 2048 u16/page → step 2048 touches one element per page.
        match &self.vectors {
            VectorStore::F16(v) => {
                for &x in v.iter().step_by(2048) { s = s.wrapping_add(x as u64); }
            }
            VectorStore::F32(v) => {
                // 1 f32 = 4 bytes → 1024 f32/page.
                for &x in v.iter().step_by(1024) { s = s.wrapping_add(x.to_bits() as u64); }
            }
        }
        // 1 u8 = 4096 u8/page.
        for &x in self.labels.iter().step_by(4096) { s = s.wrapping_add(x as u64); }
        // touch IVF data too
        if let Some(ivf) = &self.ivf {
            for &x in ivf.centroids_f32.iter().step_by(1024) { s = s.wrapping_add(x.to_bits() as u64); }
            for &x in ivf.offsets.iter().step_by(1024) { s = s.wrapping_add(x as u64); }
        }
        s
    }
}
