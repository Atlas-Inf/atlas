// SPDX-License-Identifier: AGPL-3.0-only

//! Optional numeric summaries for the determinism tracer's typed float stages.

#[derive(Default, Debug, PartialEq, Eq)]
pub(super) struct Counts {
    pub(super) nan: usize,
    pub(super) inf: usize,
    pub(super) values: usize,
    pub(super) first: Option<usize>,
}

pub(super) fn width(stage: &str, element_bytes: Option<usize>) -> Option<usize> {
    if let Some(bytes) = element_bytes {
        return matches!(bytes, 2 | 4).then_some(bytes);
    }
    match stage {
        "rt_w" => Some(4),
        "x_gate" | "emb" | "in" | "kv_lat" | "x_q_lat" | "x_q_abs" | "x_attn_lat" | "attn"
        | "attn_red" | "moe_in" | "moe_local" | "moe_red" | "moe_sh" | "moe" | "final"
        | "logits" | "plogits" | "x_qkv" | "x_g" | "x_gated" | "x_ikeys" | "x_igates" => Some(2),
        _ => None,
    }
}

impl Counts {
    pub(super) fn update(&mut self, bytes: &[u8], width: usize) {
        for word in bytes.chunks_exact(width) {
            let bits = if width == 2 {
                u32::from(u16::from_le_bytes([word[0], word[1]])) << 16
            } else {
                u32::from_le_bytes(word.try_into().expect("four-byte float"))
            };
            let value = f32::from_bits(bits);
            if !value.is_finite() {
                self.first.get_or_insert(self.values);
                self.nan += usize::from(value.is_nan());
                self.inf += usize::from(value.is_infinite());
            }
            self.values += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_counts_preserve_first_bad_position_across_segments() {
        for width in [2, 4] {
            let encode = |values: &[f32]| -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|x| {
                        let b = x.to_bits().to_le_bytes();
                        if width == 2 {
                            b[2..].to_vec()
                        } else {
                            b.to_vec()
                        }
                    })
                    .collect()
            };
            let mut c = Counts::default();
            c.update(&encode(&[0.0, -1.0, f32::INFINITY]), width);
            c.update(&encode(&[f32::NAN, f32::NEG_INFINITY, 2.0]), width);
            assert_eq!(
                c,
                Counts {
                    nan: 1,
                    inf: 2,
                    values: 6,
                    first: Some(2)
                }
            );
        }
    }

    #[test]
    fn integer_and_unknown_stages_are_not_interpreted_as_floats() {
        assert_eq!(width("rt_ids", None), None);
        assert_eq!(width("sel", None), None);
        assert_eq!(width("unknown", None), None);
        assert_eq!(width("rt_w", None), Some(4));
        assert_eq!(width("x_q_abs", None), Some(2));
    }

    #[test]
    fn highway_output_uses_its_actual_element_width() {
        assert_eq!(
            width("out", None),
            None,
            "storage is not fixed by the stage name"
        );
        for bytes in [2, 4] {
            let actual = width("out", Some(bytes)).unwrap();
            let value = f32::NAN.to_bits().to_le_bytes();
            let mut counts = Counts::default();
            counts.update(&value[4 - actual..], actual);
            assert_eq!(counts.nan, 1);
            assert_eq!(counts.values, 1);
        }
        assert_eq!(width("out", Some(1)), None);
    }
}
