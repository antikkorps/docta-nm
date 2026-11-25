//! Combinatorial sentence grammar: each relation clause a predicate has, in
//! every structural form the grammar knows.
//!
//! Four or five hand-written phrasings per predicate let a core recognise
//! strings instead of reading: it reached 84% on test and near chance on ood.
//! So each clause also takes shared frames ("Records show that ...") and a
//! parenthetical after its subject ("Varnelle, it is recorded, was ..."):
//! dozens of structures from the same words, where the subject and the value
//! never sit at fixed places.
//!
//! ood is sealed by held-out combinations: the held-out clause in every form,
//! and one (clause, form) pair in four of the seen clauses. val has its own
//! clause in every form. Every clause and every form is seen in train, never
//! all their pairings.

use crate::model::Pred;
use crate::templates::sentence_clauses;
use std::collections::BTreeSet;

/// Structural frames any clause takes; `{c}` is the clause without its stop.
pub const FRAMES: [&str; 8] = [
    "{c}.",
    "Records show that {c}.",
    "{c}, the records say.",
    "It is said that {c}.",
    "As far as anyone knows, {c}.",
    "The records say that {c}.",
    "{c}, as far as anyone knows.",
    "It is known that {c}.",
];

/// Set after a clause's opening subject, so a word never directly follows it.
pub const PARENTHETICAL: &str = ", it is recorded,";

/// One in `HELD_OUT_EVERY` (clause, form) pairs of the seen clauses is ood's.
const HELD_OUT_EVERY: usize = 4;

/// The sentence phrasings of one predicate, with `{s}` and `{o}` to fill.
#[derive(Debug, Clone)]
pub struct Sentences {
    pub seen: Vec<String>,
    pub val_held: Vec<String>,
    pub held_out: Vec<String>,
}

pub fn sentence_phrasings(pred: Pred) -> Sentences {
    let clauses = sentence_clauses(pred);
    let (mut seen, mut held_out) = (Vec::new(), Vec::new());
    for (i, clause) in clauses.seen.iter().enumerate() {
        for (j, form) in forms(clause).into_iter().enumerate() {
            if (i + j) % HELD_OUT_EVERY == HELD_OUT_EVERY - 1 {
                held_out.push(form);
            } else {
                seen.push(form);
            }
        }
    }
    held_out.extend(forms(clauses.held_out));
    // val's clause in every form; no seen pairing is taken from train, so
    // train and test are written exactly as before val had its own.
    Sentences {
        seen,
        val_held: forms(clauses.val_held),
        held_out,
    }
}

/// The capitalised words a sentence can open with that are template words,
/// never names: a copy lowers them when it repeats the sentence mid-way.
pub fn opening_words() -> BTreeSet<&'static str> {
    let clauses = Pred::ALL.into_iter().flat_map(|p| {
        let c = sentence_clauses(p);
        c.seen.iter().copied().chain([c.val_held, c.held_out])
    });
    FRAMES
        .into_iter()
        .chain(clauses)
        .filter(|t| !t.starts_with('{'))
        .filter_map(|t| t.split(' ').next())
        .collect()
}

/// A sentence repeated mid-way: an opening template word loses its capital.
pub fn lower_opening(sentence: &str) -> String {
    let first = sentence.split(' ').next().unwrap_or("");
    if opening_words().contains(first) {
        lower_first(sentence)
    } else {
        sentence.to_string()
    }
}

/// Every form of a clause: each frame, without and with the parenthetical.
fn forms(clause: &str) -> Vec<String> {
    let mut variants = vec![clause.to_string()];
    variants.extend(with_parenthetical(clause));
    variants
        .iter()
        .flat_map(|v| (0..FRAMES.len()).map(move |f| framed(v, f)))
        .collect()
}

fn framed(clause: &str, frame: usize) -> String {
    let body = clause.trim_end_matches('.');
    let template = FRAMES[frame];
    // Inside a frame the clause no longer opens the sentence.
    let body = if template.starts_with("{c}") {
        body.to_string()
    } else {
        lower_first(body)
    };
    template.replace("{c}", &body)
}

fn with_parenthetical(clause: &str) -> Option<String> {
    clause
        .strip_prefix("{s}")
        .filter(|rest| rest.starts_with(' '))
        .map(|rest| format!("{{s}}{PARENTHETICAL}{rest}"))
}

/// A name placeholder keeps its capital; an opening word loses it.
fn lower_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(c) if c != '{' => c.to_lowercase().chain(chars).collect(),
        _ => text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Pred;
    use crate::templates::sentence_clauses;

    #[test]
    fn every_predicate_reads_in_dozens_of_seen_structures() {
        for pred in Pred::ALL {
            assert!(sentence_phrasings(pred).seen.len() >= 30, "{pred:?}");
        }
    }

    #[test]
    fn seen_and_held_out_sentences_never_meet() {
        for pred in Pred::ALL {
            let s = sentence_phrasings(pred);
            let seen: BTreeSet<_> = s.seen.iter().collect();
            assert!(s.held_out.iter().all(|h| !seen.contains(h)), "{pred:?}");
            assert!(
                s.held_out
                    .contains(&sentence_clauses(pred).held_out.to_string())
            );
        }
    }

    /// A held-out combination tests composition only if both its pieces are
    /// seen: every seen clause and every form appear in train for each predicate.
    #[test]
    fn every_seen_clause_and_every_form_is_seen_for_each_predicate() {
        for pred in Pred::ALL {
            let seen = sentence_phrasings(pred).seen;
            for clause in sentence_clauses(pred).seen {
                assert!(
                    seen.iter()
                        .any(|s| s.contains(&lower_first(clause.trim_end_matches('.')))
                            || s.starts_with(clause.trim_end_matches('.'))),
                    "{pred:?}: {clause}"
                );
            }
            // The bare clause (frame "{c}.") is seen for some clause.
            assert!(
                seen.iter()
                    .any(|s| sentence_clauses(pred).seen.contains(&s.as_str())),
                "{pred:?}"
            );
            for frame in &FRAMES[1..] {
                let (head, tail) = frame.split_once("{c}").expect("a frame holds its clause");
                assert!(
                    seen.iter()
                        .any(|s| s.starts_with(head) && s.ends_with(tail)),
                    "{pred:?}: {frame}"
                );
            }
            assert!(seen.iter().any(|s| s.contains(PARENTHETICAL)), "{pred:?}");
        }
    }

    #[test]
    fn a_frame_lowers_a_clause_opening_word_but_never_a_name() {
        assert_eq!(
            framed("{s} was founded in {o}.", 1),
            "Records show that {s} was founded in {o}."
        );
        assert_eq!(
            framed("The seat of {s} is {o}.", 1),
            "Records show that the seat of {s} is {o}."
        );
        assert_eq!(
            with_parenthetical("{s} was founded in {o}."),
            Some("{s}, it is recorded, was founded in {o}.".to_string())
        );
        assert_eq!(with_parenthetical("The seat of {s} is {o}."), None);
    }
}
