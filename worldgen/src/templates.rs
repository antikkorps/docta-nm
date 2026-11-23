//! English text from templates: fills the phrasings (crate::phrasings, and
//! the sentence grammar in crate::grammar) with names and values. No LLM.

use crate::model::Pred;
pub use crate::phrasings::{
    Phrasings, comparison_phrasings, link_phrasings, question_phrasings, sentence_clauses,
};
use crate::split::Wording;
use rand::Rng;
use rand::seq::IndexedRandom;

/// A sentence stating `subj pred obj`. Seen phrasings vary at random so a model
/// cannot rely on a single surface form.
pub fn sentence<R: Rng>(
    pred: Pred,
    subj: &str,
    obj: &str,
    wording: Wording,
    rng: &mut R,
) -> String {
    let phrasings = crate::grammar::sentence_phrasings(pred);
    let pool = match wording {
        Wording::Seen => &phrasings.seen,
        Wording::ValHeld => &phrasings.val_held,
        Wording::OodHeld => &phrasings.held_out,
    };
    fill(
        pool.choose(rng).expect("templates are non-empty"),
        subj,
        obj,
    )
}

pub fn question<R: Rng>(pred: Pred, subj: &str, wording: Wording, rng: &mut R) -> String {
    fill(pick(question_phrasings(pred), wording, rng), subj, "")
}

pub fn comparison(pred: Pred, a: &str, b: &str, wording: Wording) -> String {
    let phrasings = comparison_phrasings(pred).expect("only numeric predicates are compared");
    fill(one(phrasings, wording), a, b)
}

/// Every title a source about `subj` can take. Eight forms, so a source a
/// copy cites can always get a title no other source of its world uses.
pub fn titles(subj: &str) -> Vec<String> {
    [
        "Chronicle of {s}",
        "Notes on {s}",
        "Register of {s}",
        "Annals of {s}",
        "Records of {s}",
        "Letters on {s}",
        "Survey of {s}",
        "Memoirs of {s}",
    ]
    .iter()
    .map(|t| fill(t, subj, ""))
    .collect()
}

/// Title of a source, named after the subject of its first fact.
pub fn title<R: Rng>(subj: &str, rng: &mut R) -> String {
    titles(subj).choose(rng).expect("non-empty").clone()
}

/// The first hop of a chain about `subj`, as its split words it.
pub fn link(pred: Pred, subj: &str, wording: Wording) -> String {
    let phrasings = link_phrasings(pred).expect("a chain link points to an entity");
    fill(one(phrasings, wording), subj, "")
}

/// A noun phrase for the object of `pred` of `subj`: what an unanswerable
/// question reports as missing (ground truth, never shown to a model).
pub fn describe(pred: Pred, subj: &str) -> String {
    let template = match pred {
        Pred::FoundedIn => "the founding year of {s}",
        Pred::Population => "the population of {s}",
        Pred::BornIn => "the birth year of {s}",
        Pred::Elevation => "the elevation of {s}",
        Pred::Members => "the number of members of {s}",
        Pred::Height => "the height of {s}",
        Pred::BasedIn | Pred::LivesIn | Pred::MemberOf => {
            link_phrasings(pred).expect("a link").seen[0]
        }
    };
    fill(template, subj, "")
}

/// A seen phrasing at random, or a held-out one. The held-out branches draw
/// nothing, so val and ood never shift the RNG stream of train and test.
fn pick<R: Rng>(phrasings: Phrasings, wording: Wording, rng: &mut R) -> &'static str {
    match wording {
        Wording::Seen => phrasings.seen.choose(rng).expect("templates are non-empty"),
        held => one(phrasings, held),
    }
}

/// Comparisons have one seen phrasing: no draw, so the RNG stream stays as it
/// was. The keyword baseline finds the two names joined by "or", in any wording.
fn one(phrasings: Phrasings, wording: Wording) -> &'static str {
    match wording {
        Wording::Seen => phrasings.seen[0],
        Wording::ValHeld => phrasings.val_held,
        Wording::OodHeld => phrasings.held_out,
    }
}

fn fill(template: &str, s: &str, o: &str) -> String {
    template.replace("{s}", s).replace("{o}", o)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    /// Two or three near-identical phrasings let a core recognise strings
    /// instead of reading: it scored 85% on test and chance on ood, whose
    /// structures were new though its words were all seen.
    #[test]
    fn every_predicate_reads_in_several_seen_structures() {
        for pred in Pred::ALL {
            assert!(sentence_clauses(pred).seen.len() >= 4, "{pred:?}");
            assert!(question_phrasings(pred).seen.len() >= 2, "{pred:?}");
        }
    }

    #[test]
    fn chain_questions_read_naturally() {
        let via = link(Pred::MemberOf, "Orlan Teste", Wording::Seen);
        let mut rng = rand::rngs::ChaCha8Rng::seed_from_u64(0);
        let seen = question(Pred::FoundedIn, &via, Wording::Seen, &mut rng);
        assert!(
            [
                "In what year was the guild that Orlan Teste belongs to founded?",
                "What is the founding year of the guild that Orlan Teste belongs to?",
            ]
            .contains(&seen.as_str()),
            "{seen}"
        );
        let held = link(Pred::MemberOf, "Orlan Teste", Wording::OodHeld);
        assert_eq!(
            question(Pred::FoundedIn, &held, Wording::OodHeld, &mut rng),
            "What year was the guild Orlan Teste is a member of founded in?"
        );
    }
}
