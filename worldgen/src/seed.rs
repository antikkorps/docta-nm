//! Per-world seed derivation.
//!
//! Each world gets its own seed so a world can be regenerated alone, and so
//! adding worlds at the end never changes the earlier ones.

/// Derives the seed of world `index` from the global seed.
/// It is the `index`-th output of SplitMix64 seeded with `global`, shifted to 63
/// bits so it fits a Postgres `bigint`. Collisions become possible but negligible
/// (about n²/2⁶⁴ for n worlds).
pub fn world_seed(global: u64, index: u64) -> u64 {
    const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut z = global.wrapping_add(index.wrapping_add(1).wrapping_mul(GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) >> 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic() {
        assert_eq!(world_seed(42, 7), world_seed(42, 7));
    }

    #[test]
    fn differs_across_indices_and_globals() {
        let seeds: std::collections::BTreeSet<u64> = (0..1000).map(|i| world_seed(42, i)).collect();
        assert_eq!(seeds.len(), 1000);
        assert_ne!(world_seed(42, 0), world_seed(43, 0));
    }

    #[test]
    fn fits_in_a_postgres_bigint() {
        assert!((0..10_000).all(|i| world_seed(42, i) <= i64::MAX as u64));
    }

    #[test]
    fn is_stable_across_releases() {
        // Pinned value: changing the derivation silently regenerates every dataset.
        // Reference SplitMix64 first output for seed 42, shifted to 63 bits.
        assert_eq!(world_seed(42, 0), 0xBDD7_3226_2FEB_6E95 >> 1);
    }
}
