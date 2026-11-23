//! val is written in phrasings train never sees, apart from ood's: settings
//! are chosen on how the method generalises without opening the sealed ood.

mod common;

use common::fits::{holds, skeleton};
use common::seal::{PREDS, any_fits, any_fits_loose, sentences, words};
use common::{DEFAULT, corpora_in};
use docta_worldgen::corpus::{Corpus, Primitive};
use docta_worldgen::export::seen_templates_json;
use docta_worldgen::grammar::sentence_phrasings;
use docta_worldgen::split::Split;
use docta_worldgen::templates::{
    Phrasings, comparison_phrasings, link_phrasings, question_phrasings, sentence_clauses,
};

fn val_held(of: impl Fn(docta_worldgen::model::Pred) -> Option<Phrasings>) -> Vec<&'static str> {
    PREDS
        .into_iter()
        .filter_map(of)
        .map(|p| p.val_held)
        .collect()
}

fn val_sentences() -> Vec<String> {
    PREDS
        .into_iter()
        .flat_map(|p| sentence_phrasings(p).val_held)
        .collect()
}

fn val_questions() -> Vec<&'static str> {
    let mut held = val_held(|p| Some(question_phrasings(p)));
    held.extend(val_held(comparison_phrasings));
    held
}

const OTHERS: [Split; 3] = [Split::Train, Split::Test, Split::Ood];

#[test]
fn val_sources_use_only_val_held_sentences_and_no_other_split_does() {
    let held = val_sentences();
    let held: Vec<&str> = held.iter().map(String::as_str).collect();
    for c in corpora_in(Split::Val, 100, &DEFAULT) {
        let s = sentences(&c);
        assert!(s.iter().all(|s| any_fits(&held, s)), "{}", c.world.world_id);
    }
    for c in OTHERS
        .into_iter()
        .flat_map(|s| corpora_in(s, 100, &DEFAULT))
    {
        assert!(
            sentences(&c).iter().all(|s| !any_fits(&held, s)),
            "{}",
            c.world.world_id
        );
    }
}

#[test]
fn val_questions_use_only_val_held_phrasings_and_no_other_split_does() {
    let held = val_questions();
    for c in corpora_in(Split::Val, 100, &DEFAULT) {
        assert!(
            c.questions.iter().all(|q| any_fits_loose(&held, &q.text)),
            "{}",
            c.world.world_id
        );
    }
    for c in OTHERS
        .into_iter()
        .flat_map(|s| corpora_in(s, 100, &DEFAULT))
    {
        assert!(c.questions.iter().all(|q| !any_fits_loose(&held, &q.text)));
    }
}

#[test]
fn val_chains_word_their_link_in_a_val_held_way_only() {
    let held = val_held(link_phrasings);
    let chains = |c: &Corpus| -> Vec<String> {
        c.questions
            .iter()
            .filter(|q| q.shape == Primitive::Chain)
            .map(|q| q.text.clone())
            .collect()
    };
    let val: Vec<String> = corpora_in(Split::Val, 100, &DEFAULT)
        .iter()
        .flat_map(chains)
        .collect();
    assert!(!val.is_empty());
    assert!(val.iter().all(|q| held.iter().any(|t| holds(t, q))));
    for c in OTHERS
        .into_iter()
        .flat_map(|s| corpora_in(s, 100, &DEFAULT))
    {
        assert!(chains(&c).iter().all(|q| !held.iter().any(|t| holds(t, q))));
    }
}

/// Distinct from what train sees and from what ood holds: val says something
/// about generalisation only if its phrasings are new, and ood stays sealed
/// only if none of its phrasings is spent on val.
#[test]
fn val_held_phrasings_are_neither_seen_nor_ood_held() {
    let kinds: [fn(docta_worldgen::model::Pred) -> Option<Phrasings>; 4] = [
        |p| Some(sentence_clauses(p)),
        |p| Some(question_phrasings(p)),
        comparison_phrasings,
        link_phrasings,
    ];
    for of in kinds {
        for p in PREDS.into_iter().filter_map(of) {
            assert!(
                !p.seen.contains(&p.val_held) && p.val_held != p.held_out,
                "{}",
                p.val_held
            );
        }
    }
    for pred in PREDS {
        let s = sentence_phrasings(pred);
        assert!(
            s.val_held
                .iter()
                .all(|v| !s.seen.contains(v) && !s.held_out.contains(v))
        );
    }
}

/// As for ood: new structures from words train already reads.
#[test]
fn val_held_phrasings_use_only_seen_words() {
    let seen_json = seen_templates_json();
    let seen = words([seen_json.as_str()]);
    let held = [val_questions(), val_held(link_phrasings)].concat();
    let sents = val_sentences();
    let all = held.into_iter().chain(sents.iter().map(String::as_str));
    let unseen: Vec<_> = words(all).difference(&seen).cloned().collect();
    assert!(
        unseen.is_empty(),
        "val-held words no seen phrasing uses: {unseen:?}"
    );
}

#[test]
fn every_val_held_structure_is_seen_for_another_predicate_of_its_kind() {
    for pred in PREDS {
        let held = skeleton(sentence_clauses(pred).val_held);
        let found = PREDS.into_iter().any(|other| {
            other != pred
                && other.target().is_none() == pred.target().is_none()
                && sentence_clauses(other)
                    .seen
                    .iter()
                    .any(|t| skeleton(t) == held)
        });
        assert!(
            found,
            "{pred:?}: {held:?} is seen for no other predicate of its kind"
        );
    }
}

/// The rules read the exported seen phrasings: a val-held one there would let
/// a rule pass val for free.
#[test]
fn no_val_held_phrasing_is_exported() {
    let json = seen_templates_json();
    for v in val_sentences()
        .iter()
        .map(String::as_str)
        .chain(val_questions())
    {
        assert!(!json.contains(&format!("\"{v}\"")), "{v}");
    }
    for v in val_held(link_phrasings) {
        assert!(!json.contains(v), "{v}");
    }
}
