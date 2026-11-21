//! English text templates. No LLM: every phrasing is written here by hand.
//!
//! Each kind of text has phrasings every split sees, and one held out for the
//! ood split only. A held-out phrasing is a new structure built from words the
//! seen ones already use ("In {o}, {s} was founded."): a core trained from
//! scratch learns its words from train only, so a word it never met would test
//! vocabulary, not the method. tests/ood.rs guards both.

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
                "The founding year of {s} is {o}.",
                "{o} is the year {s} was founded.",
            ],
            "In {o}, {s} was founded.",
        ),
        Pred::Population => p(
            &[
                "{s} has a population of {o}.",
                "The last census counted {o} inhabitants in {s}.",
                "{o} inhabitants live in {s}.",
                "At the last census, {s} had {o} inhabitants.",
            ],
            "The population of {s} is {o}.",
        ),
        Pred::BasedIn => p(
            &[
                "{s} is based in {o}.",
                "{s} keeps its seat in {o}.",
                "The halls of {s} stand in {o}.",
                "{o} is the seat of {s}.",
                "In {o}, {s} has its seat.",
            ],
            "The seat of {s} is {o}.",
        ),
        Pred::BornIn => p(
            &[
                "{s} was born in {o}.",
                "{s} came into the world in {o}.",
                "The birth year of {s} is {o}.",
                "In {o}, {s} was born.",
            ],
            "{o} is the year {s} was born.",
        ),
        Pred::LivesIn => p(
            &[
                "{s} lives in {o}.",
                "{s} resides in {o}.",
                "{o} is home to {s}.",
                "In {o}, {s} has a home.",
            ],
            "The home of {s} is {o}.",
        ),
        Pred::MemberOf => p(
            &[
                "{s} is a member of {o}.",
                "{s} belongs to {o}.",
                "The guild of {s} is {o}.",
                "{o} has {s} as a member.",
            ],
            "{o} is the guild of {s}.",
        ),
        Pred::Elevation => p(
            &[
                "{s} lies at an elevation of {o} metres.",
                "{s} sits {o} metres above sea level.",
                "{o} metres is the elevation of {s}.",
                "At {o} metres above sea level lies {s}.",
                "{s} lies at {o} metres above sea level.",
            ],
            "The elevation of {s} is {o} metres.",
        ),
        Pred::Members => p(
            &[
                "{s} has {o} members.",
                "The rolls of {s} list {o} members.",
                "The number of members of {s} is {o}.",
                "On the rolls of {s} are {o} members.",
            ],
            "{o} members belong to {s}.",
        ),
        Pred::Height => p(
            &[
                "{s} is {o} centimetres tall.",
                "{s} stands {o} cm tall.",
                "The height of {s} is {o} centimetres.",
                "{o} centimetres is the height of {s}.",
            ],
            "{s} stands at {o} centimetres.",
        ),
    }
}

/// Questions asking for `pred`; `{s}` may itself be a description ("the city
/// where X lives"), which is how 2-hop questions are built.
pub fn question_phrasings(pred: Pred) -> Phrasings {
    match pred {
        Pred::FoundedIn => p(
            &[
                "In what year was {s} founded?",
                "What is the founding year of {s}?",
            ],
            "What year was {s} founded in?",
        ),
        Pred::Population => p(
            &[
                "What is the population of {s}?",
                "How many people live in {s}?",
            ],
            "What population does {s} have?",
        ),
        Pred::BasedIn => p(
            &[
                "In which city is {s} based?",
                "Which city holds the seat of {s}?",
            ],
            "Which city is {s} based in?",
        ),
        Pred::BornIn => p(
            &[
                "In what year was {s} born?",
                "What is the birth year of {s}?",
            ],
            "What year was {s} born in?",
        ),
        Pred::LivesIn => p(
            &["In which city does {s} live?", "Which city is home to {s}?"],
            "Which city does {s} live in?",
        ),
        Pred::MemberOf => p(
            &[
                "Which guild does {s} belong to?",
                "Which guild has {s} as a member?",
            ],
            "Which guild is {s} a member of?",
        ),
        Pred::Elevation => p(
            &[
                "At what elevation, in metres, does {s} lie?",
                "How high above sea level does {s} lie, in metres?",
            ],
            "What is the elevation of {s}, in metres?",
        ),
        Pred::Members => p(
            &[
                "How many members does {s} have?",
                "What is the number of members of {s}?",
            ],
            "How many members belong to {s}?",
        ),
        Pred::Height => p(
            &[
                "How tall is {s}, in centimetres?",
                "What is the height of {s}, in centimetres?",
            ],
            "How tall, in centimetres, is {s}?",
        ),
    }
}

/// Two-entity comparisons; `None` for predicates that are not numeric.
pub fn comparison_phrasings(pred: Pred) -> Option<Phrasings> {
    Some(match pred {
        Pred::FoundedIn => p(
            &["Which was founded earlier, {s} or {o}?"],
            "Which of {s} or {o} was founded earlier?",
        ),
        Pred::Population => p(
            &["Which has the larger population, {s} or {o}?"],
            "Which of {s} or {o} has the larger population?",
        ),
        Pred::BornIn => p(
            &["Who was born earlier, {s} or {o}?"],
            "Which of {s} or {o} was born earlier?",
        ),
        Pred::Elevation => p(
            &["Which lies higher, {s} or {o}?"],
            "Which of {s} or {o} lies higher?",
        ),
        Pred::Members => p(
            &["Which has more members, {s} or {o}?"],
            "Which of {s} or {o} has more members?",
        ),
        Pred::Height => p(
            &["Who is taller, {s} or {o}?"],
            "Which of {s} or {o} is taller?",
        ),
        Pred::BasedIn | Pred::LivesIn | Pred::MemberOf => return None,
    })
}

/// A sentence stating `subj pred obj`. Seen phrasings vary at random so a model
/// cannot rely on a single surface form.
pub fn sentence<R: Rng>(pred: Pred, subj: &str, obj: &str, held_out: bool, rng: &mut R) -> String {
    fill(pick(sentence_phrasings(pred), held_out, rng), subj, obj)
}

pub fn question<R: Rng>(pred: Pred, subj: &str, held_out: bool, rng: &mut R) -> String {
    fill(pick(question_phrasings(pred), held_out, rng), subj, "")
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

/// How a chain names its first hop, for the predicates whose object is an
/// entity. One seen wording each (no draw, so the RNG stream stays as it was);
/// ood holds out its own, so rules that spot a chain by its seen wording do
/// not pass ood for free. Held-out words are all seen elsewhere.
pub fn link_phrasings(pred: Pred) -> Option<Phrasings> {
    match pred {
        Pred::BasedIn => Some(p(
            &["the city where {s} is based"],
            "the city {s} keeps its seat in",
        )),
        Pred::LivesIn => Some(p(&["the city where {s} lives"], "the city {s} resides in")),
        Pred::MemberOf => Some(p(
            &["the guild that {s} belongs to"],
            "the guild {s} is a member of",
        )),
        _ => None,
    }
}

/// The first hop of a chain about `subj`, as its split words it.
pub fn link(pred: Pred, subj: &str, held_out: bool) -> String {
    let phrasings = link_phrasings(pred).expect("a chain link points to an entity");
    fill(one(phrasings, held_out), subj, "")
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

/// A seen phrasing at random, or the held-out one. The held-out branch draws
/// nothing, so ood never shifts the RNG stream of the other splits.
fn pick<R: Rng>(phrasings: Phrasings, held_out: bool, rng: &mut R) -> &'static str {
    if held_out {
        phrasings.held_out
    } else {
        phrasings.seen.choose(rng).expect("templates are non-empty")
    }
}

/// Comparisons have one seen phrasing: no draw, so the RNG stream stays as it
/// was. The keyword baseline finds the two names joined by "or", in any wording.
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
    use rand::SeedableRng;

    /// Two or three near-identical phrasings let a core recognise strings
    /// instead of reading: it scored 85% on test and chance on ood, whose
    /// structures were new though its words were all seen.
    #[test]
    fn every_predicate_reads_in_several_seen_structures() {
        for pred in Pred::ALL {
            assert!(sentence_phrasings(pred).seen.len() >= 4, "{pred:?}");
            assert!(question_phrasings(pred).seen.len() >= 2, "{pred:?}");
        }
    }

    #[test]
    fn chain_questions_read_naturally() {
        let via = link(Pred::MemberOf, "Orlan Teste", false);
        let mut rng = rand::rngs::ChaCha8Rng::seed_from_u64(0);
        let seen = question(Pred::FoundedIn, &via, false, &mut rng);
        assert!(
            [
                "In what year was the guild that Orlan Teste belongs to founded?",
                "What is the founding year of the guild that Orlan Teste belongs to?",
            ]
            .contains(&seen.as_str()),
            "{seen}"
        );
        let held = link(Pred::MemberOf, "Orlan Teste", true);
        assert_eq!(
            question(Pred::FoundedIn, &held, true, &mut rng),
            "What year was the guild Orlan Teste is a member of founded in?"
        );
    }
}
