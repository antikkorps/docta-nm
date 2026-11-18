//! English text templates. No LLM: every phrasing is written here by hand.

use crate::model::Pred;
use rand::Rng;
use rand::seq::IndexedRandom;

/// A sentence stating `subj pred obj`, in one of several phrasings so a model
/// cannot rely on a single surface form.
pub fn sentence<R: Rng>(pred: Pred, subj: &str, obj: &str, rng: &mut R) -> String {
    let variants: &[&str] = match pred {
        Pred::FoundedIn => &[
            "{s} was founded in {o}.",
            "The founding of {s} dates back to {o}.",
            "{s} has existed since {o}.",
        ],
        Pred::Population => &[
            "{s} has a population of {o}.",
            "The last census counted {o} inhabitants in {s}.",
        ],
        Pred::BasedIn => &[
            "{s} is based in {o}.",
            "{s} keeps its seat in {o}.",
            "The halls of {s} stand in {o}.",
        ],
        Pred::BornIn => &["{s} was born in {o}.", "{s} came into the world in {o}."],
        Pred::LivesIn => &[
            "{s} lives in {o}.",
            "{s} resides in {o}.",
            "{o} is home to {s}.",
        ],
        Pred::MemberOf => &["{s} is a member of {o}.", "{s} belongs to {o}."],
        Pred::Elevation => &[
            "{s} lies at an elevation of {o} metres.",
            "{s} sits {o} metres above sea level.",
        ],
        Pred::Members => &["{s} has {o} members.", "The rolls of {s} list {o} members."],
        Pred::Height => &["{s} is {o} centimetres tall.", "{s} stands {o} cm tall."],
    };
    fill(pick(variants, rng), subj, obj)
}

/// Title of a source, named after the subject of its first fact.
pub fn title<R: Rng>(subj: &str, rng: &mut R) -> String {
    let variants = [
        "Chronicle of {s}",
        "Notes on {s}",
        "Register of {s}",
        "Annals of {s}",
    ];
    fill(pick(&variants, rng), subj, "")
}

/// The question asking for `pred` of `subj`; `subj` may itself be a description
/// ("the city where X lives"), which is how 2-hop questions are built.
pub fn question(pred: Pred, subj: &str) -> String {
    let template = match pred {
        Pred::FoundedIn => "In what year was {s} founded?",
        Pred::Population => "What is the population of {s}?",
        Pred::BasedIn => "In which city is {s} based?",
        Pred::BornIn => "In what year was {s} born?",
        Pred::LivesIn => "In which city does {s} live?",
        Pred::MemberOf => "Which guild does {s} belong to?",
        Pred::Elevation => "At what elevation, in metres, does {s} lie?",
        Pred::Members => "How many members does {s} have?",
        Pred::Height => "How tall is {s}, in centimetres?",
    };
    fill(template, subj, "")
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

/// A two-entity comparison on a numeric predicate.
pub fn comparison(pred: Pred, a: &str, b: &str) -> String {
    let template = match pred {
        Pred::FoundedIn => "Which was founded earlier, {s} or {o}?",
        Pred::Population => "Which has the larger population, {s} or {o}?",
        Pred::BornIn => "Who was born earlier, {s} or {o}?",
        Pred::Elevation => "Which lies higher, {s} or {o}?",
        Pred::Members => "Which has more members, {s} or {o}?",
        Pred::Height => "Who is taller, {s} or {o}?",
        Pred::BasedIn | Pred::LivesIn | Pred::MemberOf => {
            unreachable!("only numeric predicates are compared")
        }
    };
    fill(template, a, b)
}

fn pick<'a, R: Rng>(variants: &[&'a str], rng: &mut R) -> &'a str {
    variants.choose(rng).expect("templates are non-empty")
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
            question(Pred::FoundedIn, &via),
            "In what year was the guild that Orlan Teste belongs to founded?"
        );
    }
}
