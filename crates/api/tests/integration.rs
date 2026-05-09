// Integration tests against the real ./data/references.bin file.
// Run with: cargo test --release -p api --test integration -- --ignored
//
// We mark them #[ignore] so the regular `cargo test` cycle doesn't depend
// on the 91 MB binary being present.

use std::path::PathBuf;

#[path = "../src/consts.rs"] mod consts;
#[path = "../src/dataset.rs"] mod dataset;
#[path = "../src/knn.rs"] mod knn;
#[path = "../src/vectorize.rs"] mod vectorize;

fn dataset_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); p.pop(); // crates/api -> repo root
    p.push("data");
    p.push("references.bin");
    p
}

fn run_query(body: &[u8]) -> u8 {
    let p: vectorize::Payload = serde_json::from_slice(body).unwrap();
    let q = vectorize::vectorize(&p).unwrap();
    let ds = dataset::Dataset::open(&dataset_path()).expect("open dataset");
    let _ = ds.warm_up();
    let nprobe = std::env::var("NPROBE").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    knn::fraud_count_in_top_k(&q, &ds, nprobe)
}

#[test]
#[ignore]
fn legit_example_returns_zero_frauds() {
    let body = br#"{
        "id": "tx-1329056812",
        "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
        "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
        "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
        "last_transaction": null
    }"#;
    let frauds = run_query(body);
    assert_eq!(frauds, 0, "legit example should match REGRAS_DE_DETECCAO.md (0/5)");
}

#[test]
#[ignore]
fn fraud_example_returns_five_frauds() {
    let body = br#"{
        "id": "tx-3330991687",
        "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
        "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
        "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
        "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
        "last_transaction": null
    }"#;
    let frauds = run_query(body);
    assert_eq!(frauds, 5, "fraud example should match REGRAS_DE_DETECCAO.md (5/5)");
}
