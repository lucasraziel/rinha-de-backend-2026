// Payload → [f32; 14] feature vector. Follows REGRAS_DE_DETECCAO.md exactly.
// Indices 5 and 6 receive the sentinel -1.0 when last_transaction == null.

use crate::consts::*;

pub const DIMS: usize = 14;
pub const STRIDE: usize = 16;

#[derive(serde::Deserialize)]
pub struct Payload<'a> {
    #[serde(borrow, default)]
    pub id: &'a str,
    pub transaction: Transaction<'a>,
    pub customer: Customer<'a>,
    pub merchant: Merchant<'a>,
    pub terminal: Terminal,
    #[serde(default)]
    pub last_transaction: Option<LastTransaction<'a>>,
}

#[derive(serde::Deserialize)]
pub struct Transaction<'a> {
    pub amount: f32,
    pub installments: f32,
    pub requested_at: &'a str,
}

#[derive(serde::Deserialize)]
pub struct Customer<'a> {
    pub avg_amount: f32,
    pub tx_count_24h: f32,
    #[serde(borrow)]
    pub known_merchants: Vec<&'a str>,
}

#[derive(serde::Deserialize)]
pub struct Merchant<'a> {
    pub id: &'a str,
    pub mcc: &'a str,
    pub avg_amount: f32,
}

#[derive(serde::Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(serde::Deserialize)]
pub struct LastTransaction<'a> {
    pub timestamp: &'a str,
    pub km_from_current: f32,
}

#[inline(always)]
fn clamp01(x: f32) -> f32 {
    x.max(0.0).min(1.0)
}

/// Parses an ISO-8601 UTC timestamp `YYYY-MM-DDTHH:MM:SSZ` into the
/// `(hour, minute, day_of_week)` triple needed by the vectorizer.
/// `day_of_week`: Mon=0 … Sun=6 (per REGRAS_DE_DETECCAO.md).
#[inline(always)]
fn parse_iso(s: &str) -> Option<(u8, u8, u8, i64)> {
    let b = s.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[7] != b'-' || b[10] != b'T' || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let year = parse_u32(&b[0..4])? as i32;
    let month = parse_u32(&b[5..7])? as u8;
    let day = parse_u32(&b[8..10])? as u8;
    let hour = parse_u32(&b[11..13])? as u8;
    let minute = parse_u32(&b[14..16])? as u8;
    let second = parse_u32(&b[17..19])? as u8;

    let dow = day_of_week(year, month, day);

    // Days since 1970-01-01 (Unix epoch). Used only for delta in minutes.
    let days = days_from_civil(year, month, day);
    let epoch_seconds = days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64;

    Some((hour, minute, dow, epoch_seconds))
}

#[inline(always)]
fn parse_u32(b: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &c in b {
        if !c.is_ascii_digit() { return None; }
        acc = acc * 10 + (c - b'0') as u32;
    }
    Some(acc)
}

/// Mon=0 … Sun=6. Sakamoto's method.
#[inline(always)]
fn day_of_week(y: i32, m: u8, d: u8) -> u8 {
    static T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let mut y = y;
    if m < 3 { y -= 1; }
    let h = (y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d as i32) % 7;
    // Sakamoto returns Sun=0 … Sat=6. We want Mon=0 … Sun=6.
    ((h + 6) % 7) as u8
}

/// Days since civil 1970-01-01 (works for any Gregorian date).
/// Reference: Howard Hinnant's algorithm.
#[inline(always)]
fn days_from_civil(y: i32, m: u8, d: u8) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32; // [0, 399]
    let m = m as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d as u32 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    (era as i64) * 146_097 + (doe as i64) - 719_468
}

#[derive(Debug)]
pub struct VectorizeError;

#[inline]
pub fn vectorize(p: &Payload<'_>) -> Result<[f32; STRIDE], VectorizeError> {
    let mut v = [0f32; STRIDE];

    // 0  amount
    v[0] = clamp01(p.transaction.amount / MAX_AMOUNT);
    // 1  installments
    v[1] = clamp01(p.transaction.installments / MAX_INSTALLMENTS);
    // 2  amount_vs_avg
    if p.customer.avg_amount <= 0.0 {
        v[2] = 1.0;
    } else {
        v[2] = clamp01((p.transaction.amount / p.customer.avg_amount) / AMOUNT_VS_AVG_RATIO);
    }
    // 3, 4  hour_of_day, day_of_week
    let (hour, _minute, dow, now_seconds) = parse_iso(p.transaction.requested_at)
        .ok_or(VectorizeError)?;
    v[3] = (hour as f32) / 23.0;
    v[4] = (dow as f32) / 6.0;

    // 5, 6  minutes_since_last_tx, km_from_last_tx (sentinel -1 when null)
    if let Some(last) = p.last_transaction.as_ref() {
        let (_, _, _, last_seconds) = parse_iso(last.timestamp).ok_or(VectorizeError)?;
        let delta_minutes = ((now_seconds - last_seconds).max(0) as f32) / 60.0;
        v[5] = clamp01(delta_minutes / MAX_MINUTES);
        v[6] = clamp01(last.km_from_current / MAX_KM);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }

    // 7  km_from_home
    v[7] = clamp01(p.terminal.km_from_home / MAX_KM);
    // 8  tx_count_24h
    v[8] = clamp01(p.customer.tx_count_24h / MAX_TX_COUNT_24H);
    // 9  is_online
    v[9] = if p.terminal.is_online { 1.0 } else { 0.0 };
    // 10 card_present
    v[10] = if p.terminal.card_present { 1.0 } else { 0.0 };
    // 11 unknown_merchant (1 = unknown)
    let known = p.customer.known_merchants.iter().any(|m| *m == p.merchant.id);
    v[11] = if known { 0.0 } else { 1.0 };
    // 12 mcc_risk (default 0.5 if not in table)
    v[12] = mcc_risk(p.merchant.mcc.as_bytes());
    // 13 merchant_avg_amount
    v[13] = clamp01(p.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT);

    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_of_week_known_dates() {
        // 2026-03-11 is Wednesday → Mon=0, so dow = 2.
        assert_eq!(day_of_week(2026, 3, 11), 2);
        // 2026-03-14 is Saturday → dow = 5.
        assert_eq!(day_of_week(2026, 3, 14), 5);
        // 2024-01-01 is Monday → dow = 0.
        assert_eq!(day_of_week(2024, 1, 1), 0);
        // 2023-12-31 is Sunday → dow = 6.
        assert_eq!(day_of_week(2023, 12, 31), 6);
    }

    #[test]
    fn legit_example_from_regras() {
        // tx-1329056812 from REGRAS_DE_DETECCAO.md
        let body = br#"{
            "id": "tx-1329056812",
            "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
            "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
            "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
            "last_transaction": null
        }"#;
        let p: Payload = serde_json::from_slice(body).unwrap();
        let v = vectorize(&p).unwrap();
        let expected = [0.004112, 0.16667, 0.05, 0.7826, 0.3333, -1.0, -1.0, 0.02923, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006025];
        for (i, (a, b)) in v.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 0.001, "dim {i}: got {a}, want {b}");
        }
    }

    #[test]
    fn fraud_example_from_regras() {
        // tx-3330991687 from REGRAS_DE_DETECCAO.md
        let body = br#"{
            "id": "tx-3330991687",
            "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
            "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
            "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
            "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
            "last_transaction": null
        }"#;
        let p: Payload = serde_json::from_slice(body).unwrap();
        let v = vectorize(&p).unwrap();
        // expected from REGRAS_DE_DETECCAO.md (note: amount_vs_avg saturates to 1.0)
        let expected = [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1.0, -1.0, 0.9523, 1.0, 0.0, 1.0, 1.0, 0.75, 0.005486];
        for (i, (a, b)) in v.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 0.001, "dim {i}: got {a}, want {b}");
        }
    }

    #[test]
    fn unknown_mcc_defaults_to_05() {
        let body = br#"{
            "id": "x",
            "transaction": { "amount": 1.0, "installments": 1, "requested_at": "2026-01-01T00:00:00Z" },
            "customer": { "avg_amount": 1.0, "tx_count_24h": 0, "known_merchants": [] },
            "merchant": { "id": "M", "mcc": "9999", "avg_amount": 1.0 },
            "terminal": { "is_online": true, "card_present": false, "km_from_home": 0.0 },
            "last_transaction": null
        }"#;
        let p: Payload = serde_json::from_slice(body).unwrap();
        let v = vectorize(&p).unwrap();
        assert_eq!(v[12], 0.5);
    }
}
