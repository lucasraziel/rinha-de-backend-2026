// Pre-processes resources/references.json.gz into data/references.bin.
//
// F3 layout: f16 vectors with stride = 16 (14 dims + 2 zero-padding for SIMD),
// labels packed 1-bit-per-record. Header documented in api/dataset.rs.
//
//     cargo run --release -p prep -- \
//         --input  resources/references.json.gz \
//         --output data/references.bin \
//         --dtype  f16     # f32 also supported
//
// Pre-processing is deterministic — the binary is checked into git so the
// docker image build doesn't have to run prep every time.

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
const VERSION: u32 = 1;
const DIMS: u32 = 14;
const STRIDE: u32 = 16;
const DTYPE_F16: u32 = 1;
const DTYPE_F32: u32 = 2;

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vec<f32>,
    label: String,
}

#[derive(Clone, Copy)]
enum Dtype { F16, F32 }

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut input: PathBuf = "resources/references.json.gz".into();
    let mut output: PathBuf = "data/references.bin".into();
    let mut dtype = Dtype::F16;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { input = args[i + 1].clone().into(); i += 2; }
            "--output" => { output = args[i + 1].clone().into(); i += 2; }
            "--dtype" => {
                dtype = match args[i + 1].as_str() {
                    "f16" => Dtype::F16,
                    "f32" => Dtype::F32,
                    other => { eprintln!("unknown dtype: {other}"); return ExitCode::from(2); }
                };
                i += 2;
            }
            "-h" | "--help" => {
                eprintln!("usage: prep [--input PATH] [--output PATH] [--dtype f16|f32]");
                return ExitCode::SUCCESS;
            }
            other => { eprintln!("unknown arg: {other}"); return ExitCode::from(2); }
        }
    }

    if let Some(parent) = output.parent() {
        let _ = fs::create_dir_all(parent);
    }

    match run(&input, &output, dtype) {
        Ok(_) => ExitCode::SUCCESS,
        Err(e) => { eprintln!("prep: {e}"); ExitCode::from(1) }
    }
}

fn run(input: &PathBuf, output: &PathBuf, dtype: Dtype) -> io::Result<()> {
    let started = Instant::now();
    eprintln!("prep: reading {}", input.display());
    let file = File::open(input)?;
    let total_size = file.metadata()?.len();
    let gz = GzDecoder::new(BufReader::with_capacity(64 * 1024, file));

    let mut de = serde_json::Deserializer::from_reader(StreamingGz(gz));
    let records = de.deserialize_seq(SeqVisitor::default())?;

    let count = records.labels.len() as u32;
    let dtype_code = match dtype { Dtype::F16 => DTYPE_F16, Dtype::F32 => DTYPE_F32 };
    let bytes_per_lane = match dtype { Dtype::F16 => 2, Dtype::F32 => 4 };

    eprintln!(
        "prep: writing {} ({count} records, stride={STRIDE}, dtype={})",
        output.display(),
        match dtype { Dtype::F16 => "f16", Dtype::F32 => "f32" }
    );
    let out = File::create(output)?;
    let mut w = BufWriter::with_capacity(1 << 20, out);

    // header (32 bytes)
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&count.to_le_bytes())?;
    w.write_all(&DIMS.to_le_bytes())?;
    w.write_all(&STRIDE.to_le_bytes())?;
    w.write_all(&dtype_code.to_le_bytes())?;
    w.write_all(&[0u8; 8])?;

    // vectors: 14 dims + 2 zero pad each
    let mut written: u64 = 0;
    for chunk in records.vectors.chunks(DIMS as usize) {
        for &v in chunk {
            match dtype {
                Dtype::F16 => w.write_all(&f16::from_f32(v).to_bits().to_le_bytes())?,
                Dtype::F32 => w.write_all(&v.to_le_bytes())?,
            }
        }
        // pad to STRIDE with zeros
        let pad = (STRIDE as usize - DIMS as usize) * bytes_per_lane;
        w.write_all(&vec![0u8; pad])?;
        written += STRIDE as u64 * bytes_per_lane as u64;
    }

    // labels packed 1 bit per record
    let bits_len = (records.labels.len() + 7) / 8;
    let mut bits = vec![0u8; bits_len];
    for (i, &l) in records.labels.iter().enumerate() {
        if l { bits[i >> 3] |= 1 << (i & 7); }
    }
    w.write_all(&bits)?;
    w.flush()?;

    let fraud = records.labels.iter().filter(|&&l| l).count();
    let elapsed = started.elapsed();
    let total_out = 32 + written + bits_len as u64;
    eprintln!(
        "prep: done in {:.2}s — {} records ({} fraud, {} legit), input={} MB, output={} MB",
        elapsed.as_secs_f32(),
        count,
        fraud,
        count as usize - fraud,
        total_size / 1024 / 1024,
        total_out / 1024 / 1024,
    );
    drop(records);
    Ok(())
}

#[derive(Default)]
struct StreamingRecords {
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
            vectors: Vec::with_capacity(hint * DIMS as usize),
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
            out.labels.push(rec.label == "fraud");
            count += 1;
            if count % 500_000 == 0 {
                eprintln!("prep: {count} records…");
            }
        }
        Ok(out)
    }
}
