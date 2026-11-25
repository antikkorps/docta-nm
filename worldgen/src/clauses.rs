//! The relation clauses: how a source states one fact, before the grammar
//! frames them (crate::grammar). No LLM: every clause is written here by hand.
//!
//! Palier 1 of the richer language: each relation in eight or more seen
//! clauses, many of their words found nowhere else ("established", "souls",
//! "fellows"), so a core cannot learn a handful of strings by heart. Some
//! describe their subject on the way ("a figure of some renown"): words that
//! state no fact and imply none (no size, age, place, trade or tie, which a
//! value or another source could contradict), only where the predicate fixes
//! the subject's kind (a founding year belongs to a city or a guild).
//! No two relations share their distinctive words: "fellows" and "ranks"
//! count a guild's members, never say who belongs to it. The
//! clauses held out for val and for ood are unchanged: new structures built
//! from seen words (tests/ood.rs, tests/val_held.rs).

use crate::model::Pred;
use crate::phrasings::{Phrasings, p};

/// The relation clauses of `pred`, before the grammar gives each its forms
/// (crate::grammar).
pub fn sentence_clauses(pred: Pred) -> Phrasings {
    match pred {
        Pred::FoundedIn => p(
            &[
                "{s} was founded in {o}.",
                "The founding of {s} dates back to {o}.",
                "{s} has existed since {o}.",
                "The founding year of {s} is {o}.",
                "{o} is the year {s} was founded.",
                "{s} was established in {o}.",
                "{s} came into being in {o}.",
                "{s} was first recorded in {o}.",
                "{s} was set up in {o}.",
            ],
            "In {o}, {s} was founded.",
            "{s} has a founding year of {o}.",
        ),
        Pred::Population => p(
            &[
                "{s} has a population of {o}.",
                "The last census counted {o} inhabitants in {s}.",
                "{o} inhabitants live in {s}.",
                "At the last census, {s} had {o} inhabitants.",
                "{s} numbers {o} inhabitants.",
                "{s} counts some {o} souls.",
                "{s} counts {o} people.",
                "The headcount of {s} is {o}.",
                "{s}, a town of some renown, counts {o} people.",
                "{s}, a place well known to travellers, numbers {o} inhabitants.",
            ],
            "The population of {s} is {o}.",
            "{s} has {o} inhabitants.",
        ),
        Pred::BasedIn => p(
            &[
                "{s} is based in {o}.",
                "{s} keeps its seat in {o}.",
                "The halls of {s} stand in {o}.",
                "{o} is the seat of {s}.",
                "In {o}, {s} has its seat.",
                "{s} has its headquarters in {o}.",
                "The base of {s} lies in {o}.",
                "{s} meets in the great hall of {o}.",
                "{s}, a body of long standing, is based in {o}.",
                "{s}, a company well known to travellers, keeps its seat in {o}.",
            ],
            "The seat of {s} is {o}.",
            "{s} sits in {o}.",
        ),
        Pred::BornIn => p(
            &[
                "{s} was born in {o}.",
                "{s} came into the world in {o}.",
                "The birth year of {s} is {o}.",
                "In {o}, {s} was born.",
                "{s} first saw the light of day in {o}.",
                "{s} was brought into the world in {o}.",
                "The year {o} saw the birth of {s}.",
                "{s} entered the world in {o}.",
                "{s}, a name found in many records, was born in {o}.",
                "{s}, well known in the region, came into the world in {o}.",
            ],
            "{o} is the year {s} was born.",
            "The birth of {s} dates back to {o}.",
        ),
        Pred::LivesIn => p(
            &[
                "{s} lives in {o}.",
                "{s} resides in {o}.",
                "{o} is home to {s}.",
                "In {o}, {s} has a home.",
                "{s} dwells in {o}.",
                "{s} has settled in {o}.",
                "{s} makes a home in {o}.",
                "{s} is a resident of {o}.",
                "{s}, a figure of some renown, lives in {o}.",
                "{s}, often named in the songs of the region, has settled in {o}.",
            ],
            "The home of {s} is {o}.",
            "{o} is the home of {s}.",
        ),
        Pred::MemberOf => p(
            &[
                "{s} is a member of {o}.",
                "{s} belongs to {o}.",
                "The guild of {s} is {o}.",
                "{o} has {s} as a member.",
                "{s} has joined {o}.",
                "{s} is enrolled in {o}.",
                "{s} is affiliated with {o}.",
                "{s} holds a place in {o}.",
                "{s}, a figure of some renown, is a member of {o}.",
                "{s}, well known in the region, belongs to {o}.",
            ],
            "{o} is the guild of {s}.",
            "{o} is guild to {s}.",
        ),
        Pred::Elevation => p(
            &[
                "{s} lies at an elevation of {o} metres.",
                "{s} sits {o} metres above sea level.",
                "{o} metres is the elevation of {s}.",
                "At {o} metres above sea level lies {s}.",
                "{s} lies at {o} metres above sea level.",
                "{s} has an altitude of {o} metres.",
                "The altitude of {s} is {o} metres.",
                "{s} rises {o} metres above the sea.",
                "{s}, a town of some renown, lies at {o} metres above sea level.",
                "{s}, a place well known to travellers, rises {o} metres above the sea.",
            ],
            "The elevation of {s} is {o} metres.",
            "{s} stands {o} metres above sea level.",
        ),
        Pred::Members => p(
            &[
                "{s} has {o} members.",
                "The rolls of {s} list {o} members.",
                "The number of members of {s} is {o}.",
                "On the rolls of {s} are {o} members.",
                "{s} counts {o} fellows.",
                "{o} fellows fill the ranks of {s}.",
                "{s} has {o} affiliates.",
                "The ranks of {s} hold {o} fellows.",
                "{s}, a body of long standing, has {o} members.",
                "{s}, often named in the songs of the region, counts {o} fellows.",
            ],
            "{o} members belong to {s}.",
            "At the last census, {s} had {o} members.",
        ),
        Pred::Height => p(
            &[
                "{s} is {o} centimetres tall.",
                "{s} stands {o} cm tall.",
                "The height of {s} is {o} centimetres.",
                "{o} centimetres is the height of {s}.",
                "{s} measures {o} centimetres.",
                "{s} reaches a height of {o} centimetres.",
                "{s} has a height of {o} centimetres.",
                "{s} measures {o} cm from head to foot.",
                "{s}, a name found in many records, is {o} centimetres tall.",
                "{s}, often named in the songs of the region, measures {o} centimetres.",
            ],
            "{s} stands at {o} centimetres.",
            "At {o} centimetres stands {s}.",
        ),
    }
}
