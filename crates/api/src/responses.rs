// Pre-computed JSON responses for every (frauds_in_top_5, approved) pair.
// Threshold per REGRAS_DE_DETECCAO.md: approved = fraud_score < 0.6.
//
//   frauds | fraud_score | approved
//   -------+-------------+---------
//     0    | 0.0         | true
//     1    | 0.2         | true
//     2    | 0.4         | true
//     3    | 0.6         | false
//     4    | 0.8         | false
//     5    | 1.0         | false

pub const RESPONSES: [&[u8]; 6] = [
    br#"{"approved":true,"fraud_score":0.0}"#,
    br#"{"approved":true,"fraud_score":0.2}"#,
    br#"{"approved":true,"fraud_score":0.4}"#,
    br#"{"approved":false,"fraud_score":0.6}"#,
    br#"{"approved":false,"fraud_score":0.8}"#,
    br#"{"approved":false,"fraud_score":1.0}"#,
];

#[inline(always)]
pub fn response(frauds: u8) -> &'static [u8] {
    let i = if frauds as usize >= RESPONSES.len() { RESPONSES.len() - 1 } else { frauds as usize };
    RESPONSES[i]
}
