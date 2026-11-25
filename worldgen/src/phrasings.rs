//! English text templates. No LLM: every phrasing is written here by hand.
//!
//! Each kind of text has phrasings train and test see, one held out for val
//! only and one held out for the ood split only. A held-out phrasing is a new
//! structure built from words the seen ones already use ("In {o}, {s} was
//! founded."): a core trained from scratch learns its words from train only,
//! so a word it never met would test vocabulary, not the method.
//! tests/ood.rs and tests/val_held.rs guard them.

use crate::model::Pred;

/// The phrasings of one kind of text for one predicate.
#[derive(Debug, Clone, Copy)]
pub struct Phrasings {
    pub seen: &'static [&'static str],
    pub held_out: &'static str,
    /// val's own: never in train, test or ood, so val measures generalisation
    /// and settings can be chosen there without opening the sealed ood.
    pub val_held: &'static str,
}

pub(crate) const fn p(
    seen: &'static [&'static str],
    held_out: &'static str,
    val_held: &'static str,
) -> Phrasings {
    Phrasings {
        seen,
        held_out,
        val_held,
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
                "In what year was {s} established?",
            ],
            "What year was {s} founded in?",
            "What was the founding year of {s}?",
        ),
        Pred::Population => p(
            &[
                "What is the population of {s}?",
                "How many people live in {s}?",
                "How many souls does {s} count?",
            ],
            "What population does {s} have?",
            "How many inhabitants does {s} have?",
        ),
        Pred::BasedIn => p(
            &[
                "In which city is {s} based?",
                "Which city holds the seat of {s}?",
                "In which city does {s} have its headquarters?",
            ],
            "Which city is {s} based in?",
            "Which city is the seat of {s}?",
        ),
        Pred::BornIn => p(
            &[
                "In what year was {s} born?",
                "What is the birth year of {s}?",
                "In what year did {s} enter the world?",
            ],
            "What year was {s} born in?",
            "What was the birth year of {s}?",
        ),
        Pred::LivesIn => p(
            &[
                "In which city does {s} live?",
                "Which city is home to {s}?",
                "In which city does {s} dwell?",
            ],
            "Which city does {s} live in?",
            "Which city is the home of {s}?",
        ),
        Pred::MemberOf => p(
            &[
                "Which guild does {s} belong to?",
                "Which guild has {s} as a member?",
                "Which guild has {s} joined?",
            ],
            "Which guild is {s} a member of?",
            "What is the guild of {s}?",
        ),
        Pred::Elevation => p(
            &[
                "At what elevation, in metres, does {s} lie?",
                "How high above sea level does {s} lie, in metres?",
                "How many metres above the sea does {s} rise?",
            ],
            "What is the elevation of {s}, in metres?",
            "At what elevation does {s} lie, in metres?",
        ),
        Pred::Members => p(
            &[
                "How many members does {s} have?",
                "What is the number of members of {s}?",
                "How many fellows does {s} count?",
            ],
            "How many members belong to {s}?",
            "How many members are on the rolls of {s}?",
        ),
        Pred::Height => p(
            &[
                "How tall is {s}, in centimetres?",
                "What is the height of {s}, in centimetres?",
                "How many centimetres does {s} measure?",
            ],
            "How tall, in centimetres, is {s}?",
            "How many centimetres tall is {s}?",
        ),
    }
}

/// Two-entity comparisons; `None` for predicates that are not numeric.
pub fn comparison_phrasings(pred: Pred) -> Option<Phrasings> {
    Some(match pred {
        Pred::FoundedIn => p(
            &["Which was founded earlier, {s} or {o}?"],
            "Which of {s} or {o} was founded earlier?",
            "Of {s} or {o}, which was founded earlier?",
        ),
        Pred::Population => p(
            &["Which has the larger population, {s} or {o}?"],
            "Which of {s} or {o} has the larger population?",
            "Of {s} or {o}, which has the larger population?",
        ),
        Pred::BornIn => p(
            &["Who was born earlier, {s} or {o}?"],
            "Which of {s} or {o} was born earlier?",
            "Of {s} or {o}, who was born earlier?",
        ),
        Pred::Elevation => p(
            &["Which lies higher, {s} or {o}?"],
            "Which of {s} or {o} lies higher?",
            "Of {s} or {o}, which lies higher?",
        ),
        Pred::Members => p(
            &["Which has more members, {s} or {o}?"],
            "Which of {s} or {o} has more members?",
            "Of {s} or {o}, which has more members?",
        ),
        Pred::Height => p(
            &["Who is taller, {s} or {o}?"],
            "Which of {s} or {o} is taller?",
            "Of {s} or {o}, who is taller?",
        ),
        Pred::BasedIn | Pred::LivesIn | Pred::MemberOf => return None,
    })
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
            "the city that holds the seat of {s}",
        )),
        Pred::LivesIn => Some(p(
            &["the city where {s} lives"],
            "the city {s} resides in",
            "the city that is home to {s}",
        )),
        Pred::MemberOf => Some(p(
            &["the guild that {s} belongs to"],
            "the guild {s} is a member of",
            "the guild that has {s} as a member",
        )),
        _ => None,
    }
}
