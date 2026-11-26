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
/// the collisions most likely to leak real-world knowledge into a world. It
/// also holds the template words the syllables can spell: the core would read
/// a name such as "Live" as the word `live` (tests check the templates).
const EXCLUDED: &[&str] = &[
    "baker", "baku", "bali", "baron", "base", "bern", "boston", "dakar", "dallas", "data", "dates",
    "delhi", "dispute", "doha", "figure", "java", "kiev", "laos", "larger", "lemon", "level",
    "lima", "linen", "live", "lives", "lyon", "makes", "malta", "mars", "mason", "master", "melon",
    "milan", "minsk", "more", "name", "nepal", "nice", "note", "notes", "oslo", "paris", "pasta",
    "person", "peru", "register", "resides", "riga", "rise", "rises", "robin", "roma", "rome",
    "salad", "salon", "satin", "serves", "sofia", "some", "state", "stones", "sudan", "tales",
    "taller", "talon", "togo", "venus", "village",
    // Pronouns a source may write (crate::pronouns); shorter ones are never names.
    "them", "their", "they",
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

    /// Whether the syllable tables can spell `word` (2 or 3 syllables).
    fn spellable(word: &str, syllables: usize) -> bool {
        if word.is_empty() {
            return (2..=3).contains(&syllables);
        }
        syllables < 3
            && ONSETS.iter().any(|o| {
                word.strip_prefix(o).is_some_and(|rest| {
                    VOWELS.iter().any(|v| {
                        rest.strip_prefix(v).is_some_and(|rest| {
                            CODAS.iter().any(|c| {
                                rest.strip_prefix(c)
                                    .is_some_and(|rest| spellable(rest, syllables + 1))
                            })
                        })
                    })
                })
            })
    }

    /// The core reads a name that spells a template word as that word ("Live"
    /// would read as `live`), so such names must be excluded.
    #[test]
    fn no_name_spells_a_template_word() {
        use crate::model::Pred;
        use crate::templates::{comparison_phrasings, question_phrasings, sentence_clauses};
        let mut texts: Vec<&str> = crate::liars::ATTRIBUTIONS.to_vec();
        texts.push(crate::liars::HELD_OUT_ATTRIBUTION);
        texts.extend(crate::grammar::FRAMES);
        texts.extend(crate::grammar::PARENTHETICALS);
        texts.extend(crate::pronouns::WORDS);
        for pred in Pred::ALL {
            let mut kinds = vec![sentence_clauses(pred), question_phrasings(pred)];
            kinds.extend(comparison_phrasings(pred));
            for p in kinds {
                texts.extend(p.seen);
                texts.push(p.held_out);
                texts.push(p.val_held);
            }
        }
        let titles = crate::templates::titles("");
        texts.extend(titles.iter().map(String::as_str));
        let words = texts
            .iter()
            .flat_map(|t| t.split(|c: char| !c.is_ascii_alphabetic()));
        for w in words
            .map(str::to_ascii_lowercase)
            .filter(|w| (4..=12).contains(&w.len()))
        {
            assert!(
                !spellable(&w, 0) || !is_allowed(&w),
                "a name could spell {w:?}"
            );
        }
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
