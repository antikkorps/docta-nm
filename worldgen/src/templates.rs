//! English text templates. No LLM: every phrasing is written here by hand.
//!
//! Each kind of text has phrasings every split sees, and one held out for the
//! ood split only. Held-out phrasings avoid the seen ones' key words ("first
//! appeared" for a founding), so the ood split rewards reading the meaning,
//! not recognising a template.

use crate::model::Pred;
use rand::Rng;
use rand::seq::IndexedRandom;

/// The phrasings of one kind of text for one predicate.
#[derive(Debug, Clone, Copy)]
pub struct Phrasings {
    pub seen: &'static [&'static str],
    pub held_out: &'static str,
}

const fn p(seen: &'static [&'static str], held_out: &'static str) -> Phrasings {
    Phrasings { seen, held_out }
}

pub fn sentence_phrasings(pred: Pred) -> Phrasings {
    match pred {
        Pred::FoundedIn => p(
            &[
                "{s} was founded in {o}.",
                "The founding of {s} dates back to {o}.",
                "{s} has existed since {o}.",
            ],
            "{s} first appeared in the year {o}.",
        ),
        Pred::Population => p(
            &[
                "{s} has a population of {o}.",
                "The last census counted {o} inhabitants in {s}.",
            ],
            "{o} people call {s} home.",
        ),
        Pred::BasedIn => p(
            &[
                "{s} is based in {o}.",
                "{s} keeps its seat in {o}.",
                "The halls of {s} stand in {o}.",
            ],
            "{s} operates out of {o}.",
        ),
        Pred::BornIn => p(
            &["{s} was born in {o}.", "{s} came into the world in {o}."],
            "The birth of {s} took place in {o}.",
        ),
        Pred::LivesIn => p(
            &[
                "{s} lives in {o}.",
                "{s} resides in {o}.",
                "{o} is home to {s}.",
            ],
            "{s} has settled in {o}.",
        ),
        Pred::MemberOf => p(
            &["{s} is a member of {o}.", "{s} belongs to {o}."],
            "{s} joined the ranks of {o}.",
        ),
        Pred::Elevation => p(
            &[
                "{s} lies at an elevation of {o} metres.",
                "{s} sits {o} metres above sea level.",
            ],
            "The altitude of {s} is {o} metres.",
        ),
        Pred::Members => p(
            &["{s} has {o} members.", "The rolls of {s} list {o} members."],
            "{s} counts {o} members.",
        ),
        Pred::Height => p(
            &["{s} is {o} centimetres tall.", "{s} stands {o} cm tall."],
            "{s} measures {o} centimetres.",
        ),
    }
}

/// Questions asking for `pred`; `{s}` may itself be a description ("the city
/// where X lives"), which is how 2-hop questions are built.
pub fn question_phrasings(pred: Pred) -> Phrasings {
    match pred {
        Pred::FoundedIn => p(&["In what year was {s} founded?"], "When was {s} founded?"),
        Pred::Population => p(
            &["What is the population of {s}?"],
            "How many people live in {s}?",
        ),
        Pred::BasedIn => p(&["In which city is {s} based?"], "Where is {s} based?"),
        Pred::BornIn => p(&["In what year was {s} born?"], "When was {s} born?"),
        Pred::LivesIn => p(&["In which city does {s} live?"], "Where does {s} live?"),
        Pred::MemberOf => p(
            &["Which guild does {s} belong to?"],
            "Which guild is {s} a member of?",
        ),
        Pred::Elevation => p(
            &["At what elevation, in metres, does {s} lie?"],
            "How high above sea level is {s}, in metres?",
        ),
        Pred::Members => p(
            &["How many members does {s} have?"],
            "What is the membership of {s}?",
        ),
        Pred::Height => p(
            &["How tall is {s}, in centimetres?"],
            "What is the height of {s}, in centimetres?",
        ),
    }
}

/// Two-entity comparisons; `None` for predicates that are not numeric.
pub fn comparison_phrasings(pred: Pred) -> Option<Phrasings> {
    Some(match pred {
        Pred::FoundedIn => p(
            &["Which was founded earlier, {s} or {o}?"],
            "Of {s} and {o}, which is older?",
        ),
        Pred::Population => p(
            &["Which has the larger population, {s} or {o}?"],
            "Between {s} and {o}, which is more populous?",
        ),
        Pred::BornIn => p(
            &["Who was born earlier, {s} or {o}?"],
            "Of {s} and {o}, who is older?",
        ),
        Pred::Elevation => p(
            &["Which lies higher, {s} or {o}?"],
            "Between {s} and {o}, which lies at a greater altitude?",
        ),
        Pred::Members => p(
            &["Which has more members, {s} or {o}?"],
            "Of {s} and {o}, which guild is bigger?",
        ),
        Pred::Height => p(
            &["Who is taller, {s} or {o}?"],
            "Between {s} and {o}, who stands taller?",
        ),
        Pred::BasedIn | Pred::LivesIn | Pred::MemberOf => return None,
    })
}

/// A sentence stating `subj pred obj`. Seen phrasings vary at random so a model
/// cannot rely on a single surface form.
pub fn sentence<R: Rng>(pred: Pred, subj: &str, obj: &str, held_out: bool, rng: &mut R) -> String {
    let phrasings = sentence_phrasings(pred);
    // The held-out branch draws nothing, so seen splits keep their RNG stream.
    let template = if held_out {
        phrasings.held_out
    } else {
        phrasings.seen.choose(rng).expect("templates are non-empty")
    };
    fill(template, subj, obj)
}

pub fn question(pred: Pred, subj: &str, held_out: bool) -> String {
    fill(one(question_phrasings(pred), held_out), subj, "")
}

pub fn comparison(pred: Pred, a: &str, b: &str, held_out: bool) -> String {
    let phrasings = comparison_phrasings(pred).expect("only numeric predicates are compared");
    fill(one(phrasings, held_out), a, b)
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

/// A noun phrase for the object of `pred` of `subj`: the first hop of a chain,
/// and what an unanswerable question reports as missing.
pub fn describe(pred: Pred, subj: &str) -> String {
    let template = match pred {
        Pred::FoundedIn => "the founding year of {s}",
        Pred::Population => "the population of {s}",
        Pred::BasedIn => "the city where {s} is based",
        Pred::BornIn => "the birth year of {s}",
        Pred::LivesIn => "the city where {s} lives",
        Pred::MemberOf => "the guild that {s} belongs to",
        Pred::Elevation => "the elevation of {s}",
        Pred::Members => "the number of members of {s}",
        Pred::Height => "the height of {s}",
    };
    fill(template, subj, "")
}

/// Questions and comparisons have one seen phrasing: no draw, so the RNG
/// stream (and every seen split's bytes) stays as it was.
fn one(phrasings: Phrasings, held_out: bool) -> &'static str {
    if held_out {
        phrasings.held_out
    } else {
        phrasings.seen[0]
    }
}

fn fill(template: &str, s: &str, o: &str) -> String {
    template.replace("{s}", s).replace("{o}", o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_questions_read_naturally() {
        let via = describe(Pred::MemberOf, "Orlan Teste");
        assert_eq!(
            question(Pred::FoundedIn, &via, false),
            "In what year was the guild that Orlan Teste belongs to founded?"
        );
        assert_eq!(
            question(Pred::FoundedIn, &via, true),
            "When was the guild that Orlan Teste belongs to founded?"
        );
    }
}
