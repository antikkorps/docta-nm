//! Invented names, assembled from syllables so none matches a real word or place.

use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};
use std::collections::BTreeSet;

const ONSETS: &[&str] = &[
    "b", "br", "d", "dr", "f", "g", "gr", "k", "kr", "l", "m", "n", "p", "r", "s", "st", "t", "th",
    "v", "z",
];
const VOWELS: &[&str] = &["a", "e", "i", "o", "u", "ae", "ia", "ou"];
// Empty codas are repeated to make open syllables more frequent.
const CODAS: &[&str] = &["", "", "", "n", "r", "l", "s", "th"];

/// Real places and words the syllables can produce. Not exhaustive: it catches
/// the collisions most likely to leak real-world knowledge into a world.
const EXCLUDED: &[&str] = &[
    "baku", "bali", "baron", "bern", "boston", "dakar", "dallas", "data", "delhi", "doha", "java",
    "kiev", "laos", "lemon", "lima", "linen", "lyon", "malta", "mars", "melon", "milan", "minsk",
    "nepal", "nice", "oslo", "paris", "pasta", "peru", "riga", "robin", "roma", "rome", "salad",
    "salon", "satin", "sofia", "sudan", "talon", "togo", "venus",
];

/// Draws names for one world. Every name token is unique within the world, so a
/// full-text search on a name never hits two different entities.
#[derive(Default)]
pub struct NameGen {
    used: BTreeSet<String>,
}

impl NameGen {
    /// A single-token name for a place or an organisation.
    pub fn token<R: Rng>(&mut self, rng: &mut R) -> String {
        loop {
            let candidate = raw_token(rng);
            if (4..=12).contains(&candidate.len())
                && is_allowed(&candidate)
                && self.used.insert(candidate.clone())
            {
                return candidate;
            }
        }
    }

    /// A "First Last" person name.
    pub fn person<R: Rng>(&mut self, rng: &mut R) -> String {
        let first = self.token(rng);
        let last = self.token(rng);
        format!("{first} {last}")
    }
}

/// False for tokens that collide with real words or places.
pub fn is_allowed(token: &str) -> bool {
    !EXCLUDED.contains(&token.to_ascii_lowercase().as_str())
}

/// Two or three syllables, capitalised; may still be too short or excluded.
fn raw_token<R: Rng>(rng: &mut R) -> String {
    let syllables = rng.random_range(2..=3);
    let mut word = String::new();
    for _ in 0..syllables {
        for part in [ONSETS, VOWELS, CODAS] {
            word.push_str(part.choose(rng).expect("non-empty syllable table"));
        }
    }
    let mut chars = word.chars();
    let first = chars.next().expect("syllables are non-empty");
    first.to_ascii_uppercase().to_string() + chars.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::ChaCha8Rng;

    fn draw(seed: u64, n: usize) -> Vec<String> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut names = NameGen::default();
        (0..n).map(|_| names.token(&mut rng)).collect()
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(draw(1, 50), draw(1, 50));
        assert_ne!(draw(1, 50), draw(2, 50));
    }

    #[test]
    fn tokens_are_capitalised_ascii_words() {
        for t in draw(3, 300) {
            assert!((4..=12).contains(&t.len()), "{t}");
            assert!(t.chars().all(|c| c.is_ascii_alphabetic()), "{t}");
            assert!(t.starts_with(|c: char| c.is_ascii_uppercase()), "{t}");
            assert!(t[1..].chars().all(|c| c.is_ascii_lowercase()), "{t}");
        }
    }

    #[test]
    fn tokens_are_unique_and_allowed() {
        let tokens = draw(4, 500);
        let set: BTreeSet<_> = tokens.iter().collect();
        assert_eq!(set.len(), tokens.len());
        assert!(tokens.iter().all(|t| is_allowed(t)));
    }

    #[test]
    fn exclusion_list_rejects_real_names() {
        assert!(!is_allowed("Paris"));
        assert!(!is_allowed("lima"));
        assert!(is_allowed("Varnelle"));
    }

    #[test]
    fn person_tokens_do_not_reuse_place_tokens() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let mut names = NameGen::default();
        let places: Vec<_> = (0..50).map(|_| names.token(&mut rng)).collect();
        let person = names.person(&mut rng);
        let parts: Vec<_> = person.split(' ').collect();
        assert_eq!(parts.len(), 2, "{person}");
        assert!(parts.iter().all(|p| !places.iter().any(|t| t == p)));
    }
}
