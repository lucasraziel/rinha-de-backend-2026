#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::OnceLock;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use tokio::net::TcpListener;

mod consts;
mod dataset;
mod knn;
mod responses;
mod vectorize;

use dataset::Dataset;

static DATASET: OnceLock<&'static Dataset> = OnceLock::new();

const READY_BODY: &[u8] = b"";
const BAD_REQUEST_BODY: &[u8] = br#"{"error":"bad_request"}"#;

async fn handle(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => Ok(ok(READY_BODY, false)),
        (&Method::POST, "/fraud-score") => {
            let body = match req.into_body().collect().await {
                Ok(b) => b.to_bytes(),
                Err(_) => return Ok(bad_request()),
            };
            let payload: vectorize::Payload = match serde_json::from_slice(&body) {
                Ok(p) => p,
                Err(_) => return Ok(bad_request()),
            };
            let q = match vectorize::vectorize(&payload) {
                Ok(v) => v,
                Err(_) => return Ok(bad_request()),
            };
            let dataset = match DATASET.get() {
                Some(d) => *d,
                None => return Ok(internal_error()),
            };
            let frauds = knn::fraud_count_in_top_k(&q, dataset);
            let body = responses::response(frauds);
            Ok(ok(body, true))
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::new()))
            .unwrap()),
    }
}

#[inline(always)]
fn ok(body: &'static [u8], json: bool) -> Response<Full<Bytes>> {
    let mut b = Response::builder().status(StatusCode::OK);
    if json {
        b = b.header("content-type", "application/json");
    }
    b.body(Full::new(Bytes::from_static(body))).unwrap()
}

#[inline(always)]
fn bad_request() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from_static(BAD_REQUEST_BODY)))
        .unwrap()
}

#[inline(always)]
fn internal_error() -> Response<Full<Bytes>> {
    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .body(Full::new(Bytes::new()))
        .unwrap()
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(9000);
    let dataset_path: PathBuf = std::env::var("DATASET_PATH")
        .unwrap_or_else(|_| "/data/references.bin".to_string())
        .into();

    eprintln!("opening dataset at {}", dataset_path.display());
    let dataset = Dataset::open(&dataset_path)?;
    eprintln!(
        "dataset: count={} stride={} dtype={}",
        dataset.count(),
        dataset.stride(),
        dataset.header.dtype
    );
    let warm = dataset.warm_up();
    eprintln!("warm-up checksum: {warm}");
    let leaked: &'static Dataset = Box::leak(Box::new(dataset));
    let _ = DATASET.set(leaked);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = TcpListener::bind(addr).await?;
    eprintln!("api listening on {addr}");

    loop {
        let (stream, _) = listener.accept().await?;
        let _ = stream.set_nodelay(true);
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            let _ = ConnBuilder::new(TokioExecutor::new())
                .serve_connection(io, service_fn(handle))
                .await;
        });
    }
}
