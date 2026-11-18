//! The ood split must be sealed: its phrasings and chain combinations never
//! appear anywhere else, so success on it means the method generalises.

mod common;

use common::{DEFAULT, corpora_in, fact};
use docta_worldgen::corpus::{Corpus, Primitive};
use docta_worldgen::model::Pred;
use docta_worldgen::questions::HELD_OUT_CHAINS;
use docta_worldgen::split::Split;
use docta_worldgen::templates::{
    Phrasings, comparison_phrasings, question_phrasings, sentence_phrasings,
};

const PREDS: [Pred; 9] = [
    Pred::FoundedIn,
    Pred::Population,
    Pred::BasedIn,
    Pred::BornIn,
    Pred::LivesIn,
    Pred::MemberOf,
    Pred::Elevation,
    Pred::Members,
    Pred::Height,
]; // fmt: skip

/// Whether `text` could have been written from `template`: its literal pieces
/// appear in order, anchored at both ends.
fn fits(template: &str, text: &str) -> bool {
    let pieces: Vec<&str> = template.split(['{', '}']).step_by(2).collect();
    let (first, last) = (pieces[0], pieces[pieces.len() - 1]);
    if !text.starts_with(first) || !text.ends_with(last) || text.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for piece in &pieces[1..pieces.len() - 1] {
        match rest.find(piece) {
            Some(at) => rest = &rest[at + piece.len()..],
            None => return false,
        }
    }
    true
}

fn templates(of: impl Fn(Pred) -> Option<Phrasings>, held_out: bool) -> Vec<&'static str> {
    PREDS
        .into_iter()
        .filter_map(of)
        .flat_map(|p| {
            if held_out {
                vec![p.held_out]
            } else {
                p.seen.to_vec()
            }
        })
        .collect()
}

/// Every sentence a source writes itself. A copy repeats its origin's
/// sentence behind an attribution; tests/traps.rs seals the attributions.
fn sentences(c: &Corpus) -> Vec<String> {
    c.sources
        .iter()
        .filter(|s| s.copy_of.is_none())
        .flat_map(|s| {
            s.text
                .split(". ")
                .map(|x| x.trim_end_matches('.').to_string() + ".")
        })
        .collect()
}

fn any_fits(templates: &[&str], text: &str) -> bool {
    templates.iter().any(|t| fits(t, text))
}

#[test]
fn ood_sources_use_only_held_out_phrasings_and_others_never_do() {
    let held = templates(|p| Some(sentence_phrasings(p)), true);
    let seen = templates(|p| Some(sentence_phrasings(p)), false);
    for c in corpora_in(Split::Ood, 100, &DEFAULT) {
        assert!(
            sentences(&c).iter().all(|s| any_fits(&held, s)),
            "{}",
            c.world.world_id
        );
    }
    for c in corpora_in(Split::Train, 100, &DEFAULT) {
        for s in sentences(&c) {
            assert!(any_fits(&seen, &s) && !any_fits(&held, &s), "{s}");
        }
    }
}

#[test]
fn ood_questions_use_only_held_out_phrasings_and_others_never_do() {
    let mut held = templates(|p| Some(question_phrasings(p)), true);
    held.extend(templates(comparison_phrasings, true));
    for c in corpora_in(Split::Ood, 100, &DEFAULT) {
        assert!(
            c.questions.iter().all(|q| any_fits(&held, &q.text)),
            "{}",
            c.world.world_id
        );
    }
    for c in corpora_in(Split::Train, 100, &DEFAULT) {
        assert!(c.questions.iter().all(|q| !any_fits(&held, &q.text)));
    }
}

/// The predicate pair a chain question rests on, answerable or not.
fn chain_pair(c: &Corpus, q: &docta_worldgen::corpus::Question) -> (Pred, Pred) {
    (fact(c, &q.support[0]).pred, fact(c, &q.support[1]).pred)
}

#[test]
fn held_out_chain_combinations_appear_only_in_ood() {
    for split in [Split::Train, Split::Val, Split::Test] {
        for c in corpora_in(split, 100, &DEFAULT) {
            for q in c.questions.iter().filter(|q| q.shape == Primitive::Chain) {
                let pair = chain_pair(&c, q);
                assert!(!HELD_OUT_CHAINS.contains(&pair), "{split:?}: {}", q.text);
            }
        }
    }
    let ood = corpora_in(Split::Ood, 100, &DEFAULT);
    let pairs: Vec<_> = ood
        .iter()
        .flat_map(|c| {
            c.questions
                .iter()
                .filter(|q| q.shape == Primitive::Chain)
                .map(move |q| chain_pair(c, q))
        })
        .collect();
    assert!(
        pairs.iter().any(|p| HELD_OUT_CHAINS.contains(p)),
        "ood never asks a held-out chain"
    );
    assert!(
        pairs.iter().any(|p| !HELD_OUT_CHAINS.contains(p)),
        "ood is restricted to held-out chains"
    );
}

#[test]
fn ood_keeps_every_shape_and_its_share_of_unanswerables() {
    let all = corpora_in(Split::Ood, 300, &DEFAULT);
    let questions: Vec<_> = all.iter().flat_map(|c| &c.questions).collect();
    for shape in [Primitive::Retrieve, Primitive::Chain, Primitive::Compare] {
        assert!(questions.iter().any(|q| q.shape == shape), "{shape:?}");
    }
    let missing = questions.iter().filter(|q| !q.answerable()).count();
    assert_eq!(missing * 9, questions.len() * 3);
}
