//! Traps (JSONL v1): disagreeing sources around the fact a question asks about.
//! Each trap must be decidable as designed, and must not be spotted by a
//! shortcut (source id, reliability alone) instead of by reading.

mod common;

use common::{DEFAULT, PLAIN, corpora, render};
use docta_worldgen::corpus::{Corpus, Options, Primitive, Question, Source, Trap};
use docta_worldgen::split::Split;

const TRAPPED: Options = Options {
    trap_ratio: 1.0,
    ..DEFAULT
};

fn traps(c: &Corpus, kind: Trap) -> Vec<&Question> {
    c.questions
        .iter()
        .filter(|q| q.trap == Some(kind))
        .collect()
}

/// Sources that state the contested fact wrongly.
fn liars<'a>(c: &'a Corpus, q: &Question) -> Vec<&'a Source> {
    let fact = &q.support[0];
    c.sources
        .iter()
        .filter(|s| s.false_claims.iter().any(|(f, _)| f == fact))
        .collect()
}

fn truth<'a>(c: &'a Corpus, q: &Question) -> &'a Source {
    let fact = &q.support[0];
    let tellers: Vec<_> = c
        .sources
        .iter()
        .filter(|s| s.facts.contains(fact))
        .collect();
    assert_eq!(
        tellers.len(),
        1,
        "the contested fact keeps a single true source"
    );
    tellers[0]
}

#[test]
fn every_trap_occurs_and_none_without_a_trap_ratio() {
    let trapped = corpora(&TRAPPED);
    for kind in [Trap::Contradiction, Trap::FalseConsensus, Trap::Undecidable] {
        assert!(
            trapped.iter().any(|c| !traps(c, kind).is_empty()),
            "{kind:?}"
        );
    }
    for c in corpora(&PLAIN) {
        assert!(c.questions.iter().all(|q| q.trap.is_none()));
        assert!(
            c.sources
                .iter()
                .all(|s| s.false_claims.is_empty() && s.copy_of.is_none())
        );
    }
}

#[test]
fn a_contradiction_has_one_clearly_less_reliable_liar() {
    for c in corpora(&TRAPPED) {
        for q in traps(&c, Trap::Contradiction) {
            let (liars, truth) = (liars(&c, q), truth(&c, q));
            assert_eq!(liars.len(), 1);
            assert!(liars[0].copy_of.is_none());
            assert!(truth.reliability - liars[0].reliability >= 0.2 - 1e-9);
            assert_eq!(q.required_sources, vec![truth.id.clone()]);
            assert!(q.answerable() && q.expected_exit.is_none());
        }
    }
}

#[test]
fn a_false_consensus_is_one_origin_and_its_attributed_copies() {
    for c in corpora(&TRAPPED) {
        for q in traps(&c, Trap::FalseConsensus) {
            let liars = liars(&c, q);
            let origins: Vec<_> = liars.iter().filter(|s| s.copy_of.is_none()).collect();
            assert_eq!(origins.len(), 1);
            let origin = origins[0];
            let copies: Vec<_> = liars.iter().filter(|s| s.copy_of.is_some()).collect();
            assert!((2..=4).contains(&copies.len()));
            for copy in copies {
                assert_eq!(copy.copy_of.as_deref(), Some(origin.id.as_str()));
                assert!(copy.text.contains(&origin.title), "a copy names its origin");
            }
            // Strictly beyond the 0.05 that makes a pair undecidable.
            assert!(truth(&c, q).reliability - origin.reliability >= 0.1 - 1e-9);
            assert!(q.answerable());
        }
    }
}

#[test]
fn an_undecidable_trap_is_two_equally_reliable_opposite_sources() {
    for c in corpora(&TRAPPED) {
        for q in traps(&c, Trap::Undecidable) {
            let (liars, truth) = (liars(&c, q), truth(&c, q));
            assert_eq!(liars.len(), 1);
            assert!((truth.reliability - liars[0].reliability).abs() <= 0.05 + 1e-9);
            assert!(!q.answerable());
            assert_eq!(q.expected_exit, Some("unresolved_contradiction"));
            assert_eq!(
                (q.primitive, q.shape),
                (Primitive::SpotMissing, Primitive::Retrieve)
            );
            assert!(q.required_facts.is_empty() && q.required_sources.is_empty());
        }
    }
}

#[test]
fn a_false_claim_states_another_value_of_the_right_kind() {
    for c in corpora(&TRAPPED) {
        for s in &c.sources {
            for (fact_id, wrong) in &s.false_claims {
                let fact = common::fact(&c, fact_id);
                assert_ne!(&fact.obj, wrong);
                assert_eq!(
                    std::mem::discriminant(&fact.obj),
                    std::mem::discriminant(wrong)
                );
                assert!(s.text.contains(&render(&c, wrong)));
            }
        }
    }
}

#[test]
fn neither_source_ids_nor_reliability_alone_give_a_liar_away() {
    let (mut rank, mut liars, mut lowest) = (0.0, 0, 0);
    for c in corpora(&TRAPPED) {
        let n = c.sources.len() as f64;
        let floor = c
            .sources
            .iter()
            .map(|s| s.reliability)
            .fold(f64::MAX, f64::min);
        for (i, s) in c.sources.iter().enumerate() {
            if s.false_claims.is_empty() {
                continue;
            }
            assert!(
                s.reliability >= 0.6,
                "below every truthful source: {}",
                s.reliability
            );
            liars += 1;
            rank += i as f64 / n;
            lowest += (s.reliability == floor) as usize;
        }
    }
    assert!(liars > 0);
    let mean_rank = rank / liars as f64;
    assert!(
        (0.35..0.65).contains(&mean_rank),
        "liars sit at {mean_rank}"
    );
    let share = lowest as f64 / liars as f64;
    assert!(
        share < 0.4,
        "{share:.2} of liars are their world's least reliable source"
    );
}

#[test]
fn no_question_without_a_trap_rests_on_a_contested_fact() {
    for c in corpora(&TRAPPED) {
        let contested: Vec<&String> = c
            .sources
            .iter()
            .flat_map(|s| s.false_claims.iter().map(|(f, _)| f))
            .collect();
        for q in c.questions.iter().filter(|q| q.trap.is_none()) {
            assert!(
                q.support.iter().all(|f| !contested.contains(&f)),
                "{}",
                q.text
            );
        }
    }
}

#[test]
fn ood_copies_use_only_the_held_out_attribution_and_others_never_do() {
    use docta_worldgen::liars::{ATTRIBUTIONS, HELD_OUT_ATTRIBUTION};
    let opener = |phrasing: &str| phrasing.split("{t}").next().unwrap_or("").to_string();
    let held_out = HELD_OUT_ATTRIBUTION
        .split("{t}")
        .nth(1)
        .unwrap()
        .split("{x}")
        .next()
        .unwrap();
    for split in [Split::Train, Split::Ood] {
        let ood = split == Split::Ood;
        let corpora = common::corpora_in(split, 100, &TRAPPED);
        assert!(
            corpora
                .iter()
                .flat_map(|c| &c.sources)
                .any(|s| s.copy_of.is_some())
        );
        for c in corpora {
            for copy in c.sources.iter().filter(|s| s.copy_of.is_some()) {
                let seen = ATTRIBUTIONS
                    .iter()
                    .any(|a| copy.text.starts_with(&opener(a)));
                assert_eq!(seen, !ood, "{}", copy.text);
                assert_eq!(copy.text.contains(held_out), ood, "{}", copy.text);
            }
        }
    }
}

#[test]
fn a_cited_origin_is_the_only_source_with_its_title() {
    for c in corpora(&TRAPPED) {
        for copy in c.sources.iter().filter(|s| s.copy_of.is_some()) {
            let origin = copy.copy_of.as_deref().expect("a copy");
            let title = &c
                .sources
                .iter()
                .find(|s| s.id == origin)
                .expect("origin")
                .title;
            let same: Vec<_> = c.sources.iter().filter(|s| &s.title == title).collect();
            assert_eq!(same.len(), 1, "\"{title}\" names several sources");
        }
    }
}

#[test]
fn copies_are_not_longer_than_their_origin_by_more_than_today() {
    // Known shortcut (docs/design-challenge.md): the attribution makes every
    // copy longer than its origin. Filler sources (v1.1) close it; until then
    // this fails if the gap grows.
    let (mut copies, mut origins, mut n) = (0, 0, 0);
    for c in corpora(&TRAPPED) {
        for copy in c.sources.iter().filter(|s| s.copy_of.is_some()) {
            let origin = copy.copy_of.as_deref().expect("a copy");
            let origin = c.sources.iter().find(|s| s.id == origin).expect("origin");
            (copies, origins, n) = (copies + copy.text.len(), origins + origin.text.len(), n + 1);
        }
    }
    assert!(n > 0);
    let ratio = copies as f64 / origins as f64;
    assert!(
        ratio < 2.1,
        "copies are {ratio:.2} times as long as their origin"
    );
}

/// A copy repeats its origin's claim mid-sentence: an opening template word
/// ("Records", "It", "In") loses its capital there, a name keeps it. A capital
/// left after ", " or "that " would read wrong and would mark the copy.
#[test]
fn a_copied_claim_reads_on_mid_sentence() {
    let openers = docta_worldgen::grammar::opening_words();
    for c in common::corpora_in(Split::Train, 100, &TRAPPED)
        .into_iter()
        .chain(common::corpora_in(Split::Ood, 100, &TRAPPED))
    {
        for copy in c.sources.iter().filter(|s| s.copy_of.is_some()) {
            // The cited title ("Records of X") keeps its capitals.
            let origin = copy.copy_of.as_deref().expect("a copy");
            let title = &c
                .sources
                .iter()
                .find(|s| s.id == origin)
                .expect("its origin")
                .title;
            let text = copy.text.replace(title.as_str(), "");
            for w in text.split(|ch: char| !ch.is_ascii_alphabetic()).skip(1) {
                assert!(!openers.contains(w), "{}", copy.text);
            }
        }
    }
}
