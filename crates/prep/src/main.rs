// Pre-processes resources/references.json.gz into data/references.bin.
//
// Output layout (header version 2 with optional IVF section):
//
//     header (32 B):
//         magic "R26V"  version=2
//         count  dims=14  stride=16  dtype (1=f16, 2=f32)
//         nlist (u32, 0 if no IVF)  flags (u32, bit 0 = IVF present)
//
//     vectors:  count × stride × sizeof(dtype)        ← reordered by cluster if IVF
//     labels:   ceil(count / 8) bytes                 ← reordered to match
//     centroids (if IVF): nlist × stride × 2 (f16)
//     offsets   (if IVF): (nlist + 1) × 4 (u32)
//
//     cargo run --release -p prep -- \
//         --input  resources/references.json.gz \
//         --output data/references.bin \
//         --dtype  f16 \
//         --ivf 1024 --max-iter 30 --seed 42

mod kmeans;

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use flate2::read::GzDecoder;
use half::f16;
use serde::de::Deserializer as _;
use serde::Deserialize;

const MAGIC: &[u8; 4] = b"R26V";
const VERSION: u32 = 2;
const DIMS: u32 = 14;
const STRIDE: u32 = 16;
const DTYPE_F16: u32 = 1;
const DTYPE_F32: u32 = 2;
const FLAG_IVF: u32 = 1;

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vec<f32>,
    label: String,
}

#[derive(Clone, Copy)]
enum Dtype { F16, F32 }

struct Args {
    input: PathBuf,
    output: PathBuf,
    dtype: Dtype,
    ivf_nlist: u32,    // 0 = no IVF
    max_iter: usize,
    tol: f32,
    seed: u64,
}

fn main() -> ExitCode {
    let mut args = Args {
        input: "resources/references.json.gz".into(),
        output: "data/references.bin".into(),
        dtype: Dtype::F16,
        ivf_nlist: 1024,
        max_iter: 30,
        tol: 0.0005,
        seed: 42,
    };
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--input"    => { args.input    = argv[i + 1].clone().into(); i += 2; }
            "--output"   => { args.output   = argv[i + 1].clone().into(); i += 2; }
            "--dtype"    => {
                args.dtype = match argv[i + 1].as_str() {
                    "f16" => Dtype::F16,
                    "f32" => Dtype::F32,
                    other => { eprintln!("unknown dtype: {other}"); return ExitCode::from(2); }
                };
                i += 2;
            }
            "--ivf"      => { args.ivf_nlist = argv[i + 1].parse().unwrap(); i += 2; }
            "--max-iter" => { args.max_iter  = argv[i + 1].parse().unwrap(); i += 2; }
            "--tol"      => { args.tol       = argv[i + 1].parse().unwrap(); i += 2; }
            "--seed"     => { args.seed      = argv[i + 1].parse().unwrap(); i += 2; }
            "-h" | "--help" => {
                eprintln!(
                    "usage: prep [--input PATH] [--output PATH] [--dtype f16|f32]\n\
                                [--ivf NLIST] [--max-iter N] [--tol F] [--seed N]"
                );
                return ExitCode::SUCCESS;
            }
            other => { eprintln!("unknown arg: {other}"); return ExitCode::from(2); }
        }
    }

    if let Some(parent) = args.output.parent() {
        let _ = fs::create_dir_all(parent);
    }

    match run(&args) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("prep: {e}"); ExitCode::from(1) }
    }
}

fn run(args: &Args) -> io::Result<()> {
    let started = Instant::now();
    eprintln!("prep: reading {}", args.input.display());
    let file = File::open(&args.input)?;
    let total_size = file.metadata()?.len();
    let gz = GzDecoder::new(BufReader::with_capacity(64 * 1024, file));

    let mut de = serde_json::Deserializer::from_reader(StreamingGz(gz));
    let mut records = de.deserialize_seq(SeqVisitor::default())?;
    let count = records.labels.len();
    let stride = STRIDE as usize;

    // The streaming visitor emits stride-16 vectors (14 real dims + 2 zeros)
    // so kmeans / api can SIMD over the same buffer without reshape.
    debug_assert_eq!(records.vectors.len(), count * stride);

    let (vectors, labels, ivf) = if args.ivf_nlist > 0 {
        eprintln!(
            "prep: kmeans nlist={} max_iter={} seed={}",
            args.ivf_nlist, args.max_iter, args.seed
        );
        let km = kmeans::run(
            &records.vectors,
            count,
            args.ivf_nlist as usize,
            args.max_iter,
            args.tol,
            args.seed,
        );
        eprintln!(
            "prep: kmeans done in {} iters (final max_change {:.6})",
            km.iters, km.final_max_change
        );

        // reorder vectors and labels grouped by cluster
        let nlist = args.ivf_nlist as usize;
        let mut counts = vec![0u32; nlist];
        for &c in &km.assignments { counts[c as usize] += 1; }
        let mut offsets = vec![0u32; nlist + 1];
        for c in 0..nlist { offsets[c + 1] = offsets[c] + counts[c]; }
        let mut cursor = offsets[..nlist].to_vec();

        let mut new_vectors = vec![0f32; count * stride];
        let mut new_labels = vec![false; count];
        for i in 0..count {
            let c = km.assignments[i] as usize;
            let dst = cursor[c] as usize;
            cursor[c] += 1;
            new_vectors[dst * stride..(dst + 1) * stride]
                .copy_from_slice(&records.vectors[i * stride..(i + 1) * stride]);
            new_labels[dst] = records.labels[i];
        }

        records.vectors = Vec::new(); // free
        records.labels = Vec::new();

        (new_vectors, new_labels, Some((km.centroids, offsets)))
    } else {
        (records.vectors, records.labels, None)
    };

    let dtype_code = match args.dtype { Dtype::F16 => DTYPE_F16, Dtype::F32 => DTYPE_F32 };
    let bytes_per_lane = match args.dtype { Dtype::F16 => 2, Dtype::F32 => 4 };
    let mut flags = 0u32;
    if ivf.is_some() { flags |= FLAG_IVF; }

    eprintln!(
        "prep: writing {} ({count} records, stride={STRIDE}, dtype={}, nlist={}, flags={:#x})",
        args.output.display(),
        match args.dtype { Dtype::F16 => "f16", Dtype::F32 => "f32" },
        args.ivf_nlist,
        flags,
    );
    let out = File::create(&args.output)?;
    let mut w = BufWriter::with_capacity(1 << 20, out);

    // header (32 bytes)
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(count as u32).to_le_bytes())?;
    w.write_all(&DIMS.to_le_bytes())?;
    w.write_all(&STRIDE.to_le_bytes())?;
    w.write_all(&dtype_code.to_le_bytes())?;
    w.write_all(&args.ivf_nlist.to_le_bytes())?;
    w.write_all(&flags.to_le_bytes())?;

    // vectors
    for &v in &vectors {
        match args.dtype {
            Dtype::F16 => w.write_all(&f16::from_f32(v).to_bits().to_le_bytes())?,
            Dtype::F32 => w.write_all(&v.to_le_bytes())?,
        }
    }

    // labels packed
    let bits_len = (count + 7) / 8;
    let mut bits = vec![0u8; bits_len];
    for (i, &l) in labels.iter().enumerate() {
        if l { bits[i >> 3] |= 1 << (i & 7); }
    }
    w.write_all(&bits)?;

    // centroids + offsets
    if let Some((centroids, offsets)) = &ivf {
        for &v in centroids {
            match args.dtype {
                Dtype::F16 => w.write_all(&f16::from_f32(v).to_bits().to_le_bytes())?,
                Dtype::F32 => w.write_all(&v.to_le_bytes())?,
            }
        }
        for &o in offsets {
            w.write_all(&o.to_le_bytes())?;
        }
    }

    w.flush()?;

    let fraud = labels.iter().filter(|&&l| l).count();
    let elapsed = started.elapsed();
    let vectors_bytes = count * stride * bytes_per_lane;
    let total_out = 32 + vectors_bytes + bits_len
        + match &ivf {
            Some(_) => args.ivf_nlist as usize * stride * bytes_per_lane + (args.ivf_nlist as usize + 1) * 4,
            None => 0,
        };

    eprintln!(
        "prep: done in {:.2}s — {} records ({} fraud, {} legit), input={} MB, output={} MB",
        elapsed.as_secs_f32(),
        count,
        fraud,
        count - fraud,
        total_size / 1024 / 1024,
        total_out / 1024 / 1024,
    );
    Ok(())
}

#[derive(Default)]
struct StreamingRecords {
    /// flat: count × STRIDE f32 (already padded with 2 zeros per row)
    vectors: Vec<f32>,
    labels: Vec<bool>,
}

struct StreamingGz<R>(R);

impl<R: Read> Read for StreamingGz<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> { self.0.read(buf) }
}

#[derive(Default)]
struct SeqVisitor;

impl<'de> serde::de::Visitor<'de> for SeqVisitor {
    type Value = StreamingRecords;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("array of {vector, label}")
    }
    fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let hint = seq.size_hint().unwrap_or(3_000_000);
        let mut out = StreamingRecords {
            vectors: Vec::with_capacity(hint * STRIDE as usize),
            labels: Vec::with_capacity(hint),
        };
        let mut count = 0usize;
        while let Some(rec) = seq.next_element::<ReferenceRecord>()? {
            if rec.vector.len() != DIMS as usize {
                return Err(serde::de::Error::custom(format!(
                    "expected {} dims, got {}",
                    DIMS,
                    rec.vector.len()
                )));
            }
            out.vectors.extend_from_slice(&rec.vector);
            // pad to STRIDE with zeros
            for _ in DIMS..STRIDE { out.vectors.push(0.0); }
            out.labels.push(rec.label == "fraud");
            count += 1;
            if count % 500_000 == 0 {
                eprintln!("prep: {count} records…");
            }
        }
        Ok(out)
    }
}
