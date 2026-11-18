//! Guarantees of sources and questions, checked against the world's ground truth.

mod common;

use common::{DEFAULT, corpora, fact, name, render, sources_with};
use docta_worldgen::corpus::{Corpus, Options, Primitive, Question, build};
use docta_worldgen::model::{Fact, Obj, Pred};
use docta_worldgen::world::generate;
use std::collections::BTreeSet;

/// Re-derives the answer from the required facts alone, independently of the
/// generator's code.
fn solve(c: &Corpus, q: &Question) -> String {
    let facts: Vec<&Fact> = q.required_facts.iter().map(|id| fact(c, id)).collect();
    match (q.primitive, facts.as_slice()) {
        (Primitive::Retrieve, [f]) => render(c, &f.obj),
        (Primitive::Chain, [a, b]) => {
            assert_eq!(a.obj, Obj::Entity(b.subj.clone()), "{}: broken chain", q.id);
            render(c, &b.obj)
        }
        (Primitive::Compare, [a, b]) => compare(c, a, b),
        _ => panic!("{}: {:?} with {} facts", q.id, q.primitive, facts.len()),
    }
}

fn compare(c: &Corpus, a: &Fact, b: &Fact) -> String {
    assert_eq!(a.pred, b.pred);
    assert_ne!(a.subj, b.subj);
    let (Obj::Int(x), Obj::Int(y)) = (&a.obj, &b.obj) else {
        panic!("compare needs numbers");
    };
    // Written out rather than calling Pred::larger_wins, to stay independent.
    let a_wins = match a.pred {
        Pred::FoundedIn | Pred::BornIn => x < y, // years: the earlier one
        _ => x > y,
    };
    name(c, if a_wins { &a.subj } else { &b.subj })
}

#[test]
fn each_entity_has_exactly_one_unstated_fact_and_the_rest_once() {
    // Every entity lacks one fact in the sources, so being under-documented
    // never singles out the subject of an unanswerable question.
    for c in corpora(&DEFAULT) {
        for e in &c.world.entities {
            let stated: Vec<usize> = c
                .world
                .facts
                .iter()
                .filter(|f| f.subj == e.id)
                .map(|f| sources_with(&c, &f.id).len())
                .collect();
            assert_eq!(stated.iter().filter(|&&n| n == 0).count(), 1, "{}", e.id);
            assert!(
                stated.iter().all(|&n| n <= 1),
                "{}: fact in two sources",
                e.id
            );
        }
    }
}

#[test]
fn sources_hold_one_to_three_facts_and_state_them() {
    for c in corpora(&DEFAULT) {
        for s in &c.sources {
            assert!((1..=3).contains(&s.facts.len()), "{}", s.id);
            assert!((0.6..=1.0).contains(&s.reliability));
            for f in &s.facts {
                let f = fact(&c, f);
                for part in [name(&c, &f.subj), render(&c, &f.obj)] {
                    assert!(
                        s.text.contains(&part),
                        "{}: {part} not in {:?}",
                        s.id,
                        s.text
                    );
                }
            }
        }
    }
}

#[test]
fn answers_follow_from_required_facts_in_required_sources() {
    for c in corpora(&DEFAULT) {
        for q in c.questions.iter().filter(|q| q.answerable()) {
            assert_eq!(
                q.answer.as_deref(),
                Some(solve(&c, q).as_str()),
                "{}",
                q.text
            );
            assert_eq!(q.required_sources.len(), q.required_facts.len());
            for (f, s) in q.required_facts.iter().zip(&q.required_sources) {
                assert_eq!(sources_with(&c, f)[0].id, *s, "{}", q.id);
            }
            let hops = if q.primitive == Primitive::Chain {
                2
            } else {
                1
            };
            assert_eq!(q.hops, hops);
        }
    }
}

#[test]
fn unanswerable_questions_have_no_support_in_any_source() {
    for c in corpora(&DEFAULT) {
        for q in c.questions.iter().filter(|q| !q.answerable()) {
            assert_eq!(q.primitive, Primitive::SpotMissing);
            assert!(
                q.missing.is_some() && q.required_facts.is_empty() && q.required_sources.is_empty()
            );
            let target = fact(&c, q.absent_fact.as_ref().expect("absent fact"));
            let supported = c.sources.iter().flat_map(|s| &s.facts).any(|id| {
                let f = fact(&c, id);
                f.subj == target.subj && f.pred == target.pred
            });
            assert!(!supported, "{}: answerable from sources", q.text);
            let subject = name(&c, &target.subj);
            assert!(
                c.sources.iter().any(|s| s.text.contains(&subject)),
                "{subject} never cited"
            );
        }
    }
}

#[test]
fn question_mix_follows_the_options() {
    for c in corpora(&DEFAULT) {
        assert_eq!(c.questions.len(), 9);
        assert_eq!(c.questions.iter().filter(|q| !q.answerable()).count(), 3);
        let texts: BTreeSet<_> = c.questions.iter().map(|q| &q.text).collect();
        assert_eq!(texts.len(), 9, "duplicate question in {}", c.world.world_id);
    }
    let none = Options {
        unanswerable_ratio: 0.0,
        ..DEFAULT
    };
    assert!(
        corpora(&none)
            .iter()
            .flat_map(|c| &c.questions)
            .all(|q| q.answerable())
    );
}

#[test]
fn every_primitive_occurs_and_max_hops_is_respected() {
    let used: BTreeSet<_> = corpora(&DEFAULT)
        .iter()
        .flat_map(|c| c.questions.iter().map(|q| q.primitive.as_str()))
        .collect();
    assert_eq!(used.len(), 4, "{used:?}");
    let one_hop = Options {
        max_hops: 1,
        ..DEFAULT
    };
    assert!(
        corpora(&one_hop)
            .iter()
            .flat_map(|c| &c.questions)
            .all(|q| q.hops == 1)
    );
}

#[test]
fn build_is_deterministic() {
    let w = || generate("w", 3);
    assert_eq!(build(w(), &DEFAULT), build(w(), &DEFAULT));
}
