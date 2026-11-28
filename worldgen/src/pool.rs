//! A pool of invented names for the delexicalised corpus: real Wikipedia
//! sentences get their names replaced by these, so every name in the project
//! comes from one generator.

use crate::names::NameGen;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

/// `n` distinct single-word names drawn from `seed`. Drawn one after the other
/// from one generator, so a longer pool starts with the shorter one.
pub fn pool(seed: u64, n: usize) -> Vec<String> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut names = NameGen::default();
    (0..n).map(|_| names.token(&mut rng)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic_and_seeded() {
        assert_eq!(pool(7, 100), pool(7, 100));
        assert_ne!(pool(7, 100), pool(8, 100));
    }

    #[test]
    fn names_are_distinct_allowed_single_words() {
        let names = pool(7, 5000);
        let distinct: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(distinct.len(), 5000);
        assert!(
            names
                .iter()
                .all(|n| crate::names::is_allowed(n) && !n.contains(' '))
        );
    }

    #[test]
    fn a_longer_pool_extends_a_shorter_one() {
        // So asking for more names never renames what was already replaced.
        assert_eq!(pool(7, 1000)[..100], pool(7, 100)[..]);
    }
}
