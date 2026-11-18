//! Question-set quality: no redundancy, and no cue that gives away an
//! unanswerable question without reading the sources.

mod common;

use common::{DEFAULT, corpora, corpora_in, corpora_n};
use docta_worldgen::corpus::{Corpus, Primitive, Question};
use docta_worldgen::split::Split;
use std::collections::BTreeSet;

#[test]
fn no_two_questions_ask_the_same_thing() {
    for c in corpora(&DEFAULT) {
        let keys: BTreeSet<_> = c
            .questions
            .iter()
            .filter(|q| q.answerable())
            .map(|q| {
                let facts: BTreeSet<_> = q.required_facts.iter().collect();
                (q.primitive.as_str(), facts)
            })
            .collect();
        let answerable = c.questions.iter().filter(|q| q.answerable()).count();
        assert_eq!(
            keys.len(),
            answerable,
            "redundant questions in {}",
            c.world.world_id
        );
    }
}

/// The fact a single-subject question is about: the one it reads, or the one
/// no source states.
fn subject_of<'a>(c: &'a Corpus, q: &Question) -> &'a str {
    let id = q
        .absent_fact
        .as_ref()
        .unwrap_or_else(|| &q.required_facts[0]);
    let f = c
        .world
        .facts
        .iter()
        .find(|f| &f.id == id)
        .expect("known fact");
    &c.world.entity(&f.subj).expect("known entity").name
}

/// Mean number of sources naming the subject of single-subject (`retrieve`
/// shaped) questions, answerable or not.
fn mean_mentions(corpora: &[Corpus], answerable: bool) -> f64 {
    let counts: Vec<usize> = corpora
        .iter()
        .flat_map(|c| {
            c.questions
                .iter()
                .filter(|q| q.shape == Primitive::Retrieve && q.answerable() == answerable)
                .map(|q| {
                    let subject = subject_of(c, q);
                    c.sources
                        .iter()
                        .filter(|s| s.text.contains(subject))
                        .count()
                })
        })
        .collect();
    counts.iter().sum::<usize>() as f64 / counts.len() as f64
}

#[test]
fn unanswerable_subjects_are_as_documented_as_answerable_ones() {
    // A model sees one question with its sources. If the subject of an
    // unanswerable question were cited less, "few mentions => abstain" would
    // score without checking the predicate.
    let all = corpora_n(1000, &DEFAULT);
    let missing = mean_mentions(&all, false);
    let answerable = mean_mentions(&all, true);
    assert!(
        (missing - answerable).abs() < 0.1,
        "unanswerable {missing:.3} vs answerable {answerable:.3}"
    );
}

#[test]
fn every_question_shape_is_unanswerable_equally_often() {
    // If only some shapes could lack an answer, the question text alone would
    // predict abstention. Checked on ood too: its chain pool differs.
    for split in [Split::Train, Split::Ood] {
        let all = corpora_in(split, 1000, &DEFAULT);
        let questions: Vec<&Question> = all.iter().flat_map(|c| &c.questions).collect();
        let share = |qs: &[&&Question]| {
            qs.iter().filter(|q| !q.answerable()).count() as f64 / qs.len() as f64
        };
        let overall = share(&questions.iter().collect::<Vec<_>>());
        for shape in [Primitive::Retrieve, Primitive::Chain, Primitive::Compare] {
            let of_shape: Vec<_> = questions.iter().filter(|q| q.shape == shape).collect();
            let s = share(&of_shape);
            assert!(
                (s - overall).abs() < 0.03,
                "{split:?} {shape:?}: {s:.3} unanswerable vs {overall:.3}"
            );
        }
    }
}
