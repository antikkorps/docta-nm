//! The ood split must be sealed: its phrasings and chain combinations never
//! appear anywhere else, so success on it means the method generalises.

mod common;

use common::fits::{holds, skeleton};
use common::seal::{PREDS, any_fits, any_fits_loose, sentences, words};
use common::{DEFAULT, corpora_in, fact};
use docta_worldgen::corpus::{Corpus, Primitive};
use docta_worldgen::grammar::{sentence_phrasings, with_pronouns};
use docta_worldgen::liars::{ATTRIBUTIONS, HELD_OUT_ATTRIBUTION};
use docta_worldgen::model::Pred;
use docta_worldgen::questions::HELD_OUT_CHAINS;
use docta_worldgen::split::Split;
use docta_worldgen::templates::{
    Phrasings, comparison_phrasings, describe, link_phrasings, question_phrasings,
    sentence_clauses, titles,
};

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

/// Every sentence phrasing the grammar writes, seen or held out.
fn sentence_templates(held_out: bool) -> Vec<String> {
    PREDS
        .into_iter()
        .flat_map(|p| {
            let s = sentence_phrasings(p);
            with_pronouns(p, if held_out { s.held_out } else { s.seen })
        })
        .collect()
}

#[test]
fn ood_sources_use_only_held_out_phrasings_and_others_never_do() {
    let (held, seen) = (sentence_templates(true), sentence_templates(false));
    let held: Vec<&str> = held.iter().map(String::as_str).collect();
    let seen: Vec<&str> = seen.iter().map(String::as_str).collect();
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
            c.questions.iter().all(|q| any_fits_loose(&held, &q.text)),
            "{}",
            c.world.world_id
        );
    }
    for c in corpora_in(Split::Train, 100, &DEFAULT) {
        assert!(c.questions.iter().all(|q| !any_fits_loose(&held, &q.text)));
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

/// A core learns words from train only: an ood word it never met cannot be
/// read, whatever the method learned. Held-out phrasings recombine seen
/// words in new structures, so ood measures generalisation, not vocabulary.
#[test]
fn held_out_phrasings_use_only_words_the_seen_ones_use() {
    let seen_texts: Vec<String> = [
        templates(|p| Some(question_phrasings(p)), false),
        templates(comparison_phrasings, false),
        ATTRIBUTIONS.to_vec(),
    ]
    .concat()
    .into_iter()
    .map(String::from)
    .chain(titles("{s}"))
    .chain(sentence_templates(false))
    // Only chain links reach a model; other descriptions stay in ground truth.
    .chain(
        [Pred::BasedIn, Pred::LivesIn, Pred::MemberOf]
            .into_iter()
            .map(|p| describe(p, "{s}")),
    )
    .collect();
    let seen = words(seen_texts.iter().map(String::as_str));
    let held = [
        templates(|p| Some(question_phrasings(p)), true),
        templates(comparison_phrasings, true),
        vec![HELD_OUT_ATTRIBUTION],
        templates(link_phrasings, true),
    ]
    .concat();
    let held_sentences = sentence_templates(true);
    let held = held
        .into_iter()
        .chain(held_sentences.iter().map(String::as_str));
    let unseen: Vec<_> = words(held).difference(&seen).cloned().collect();
    assert!(
        unseen.is_empty(),
        "held-out words no seen phrasing uses: {unseen:?}"
    );
}

/// ood tests new combinations of known pieces: a held-out sentence's structure
/// must be seen for another predicate with the same kind of object (a number
/// or an entity). "In {o}, {s} was founded." stayed at chance while "In {o}"
/// was only ever seen before a city.
#[test]
fn every_held_out_structure_is_seen_for_another_predicate_of_its_kind() {
    for pred in PREDS {
        let held = skeleton(sentence_clauses(pred).held_out);
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

/// Rules that spot a chain by its seen link wording must not pass ood for
/// free: ood words its links in its own way, so a chain gain there shows a
/// decomposition that generalises, not a regex that matched.
#[test]
fn ood_chains_word_their_link_in_a_held_out_way_only() {
    let held = templates(link_phrasings, true);
    let seen = templates(link_phrasings, false);
    let chains = |c: &Corpus| -> Vec<String> {
        c.questions
            .iter()
            .filter(|q| q.shape == Primitive::Chain)
            .map(|q| q.text.clone())
            .collect()
    };
    let ood: Vec<String> = corpora_in(Split::Ood, 100, &DEFAULT)
        .iter()
        .flat_map(chains)
        .collect();
    assert!(!ood.is_empty());
    for q in &ood {
        assert!(
            held.iter().any(|t| holds(t, q)) && !seen.iter().any(|t| holds(t, q)),
            "{q}"
        );
    }
    // val words its links its own way too (tests/val_held.rs), never ood's.
    for c in corpora_in(Split::Val, 100, &DEFAULT) {
        assert!(chains(&c).iter().all(|q| !held.iter().any(|t| holds(t, q))));
    }
    let others = [Split::Train, Split::Test];
    for c in others
        .into_iter()
        .flat_map(|s| corpora_in(s, 100, &DEFAULT))
    {
        for q in chains(&c) {
            assert!(
                seen.iter().any(|t| holds(t, &q)) && !held.iter().any(|t| holds(t, &q)),
                "{q}"
            );
        }
    }
}
