//! A richer language, palier 1: each idea said in many words, so a core
//! trained from scratch cannot learn a handful of strings by heart. With 2 or
//! 3 ways per idea the Read core memorised train (train loss 0.15, val flat).

use docta_worldgen::grammar::{FRAMES, PARENTHETICALS, echoes, sentence_phrasings};
use docta_worldgen::model::Pred;
use docta_worldgen::templates::{question_phrasings, sentence_clauses, titles};
use std::collections::BTreeSet;

/// Lower-case words of a template, its placeholders left out.
fn words(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_ascii_alphabetic() && c != '{' && c != '}')
        .filter(|w| !w.is_empty() && !w.starts_with('{'))
        .map(str::to_ascii_lowercase)
}

fn seen_vocabulary() -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    for pred in Pred::ALL {
        for s in sentence_phrasings(pred).seen {
            seen.extend(words(&s));
        }
        for q in question_phrasings(pred).seen {
            seen.extend(words(q));
        }
    }
    for t in titles("{s}") {
        seen.extend(words(&t));
    }
    seen
}

#[test]
fn every_relation_is_said_in_at_least_eight_seen_clauses() {
    for pred in Pred::ALL {
        assert!(sentence_clauses(pred).seen.len() >= 8, "{pred:?}");
    }
}

#[test]
fn sources_frame_their_sentences_in_many_ways() {
    assert!(FRAMES.len() >= 16);
    assert!(PARENTHETICALS.len() >= 4);
    assert!(titles("{s}").len() >= 16);
}

#[test]
fn every_question_has_at_least_three_seen_wordings() {
    for pred in Pred::ALL {
        assert!(question_phrasings(pred).seen.len() >= 3, "{pred:?}");
    }
}

/// Before palier 1 the seen templates used about 90 distinct words.
#[test]
fn the_seen_vocabulary_has_grown_past_two_hundred_words() {
    let seen = seen_vocabulary();
    assert!(seen.len() >= 200, "{} words", seen.len());
}

/// "By all accounts, X, by all accounts, is…": an aside never shares a
/// distinctive word with its frame.
#[test]
fn no_sentence_says_a_word_of_its_aside_twice() {
    for (frame, aside) in [
        ("By all accounts, {c}.", ", by all accounts,"),
        ("{c}, so the elders say.", ", so the elders tell,"),
        ("{c}, the chronicles say.", ", as the chronicles note,"),
        ("{c}, as the archives note.", ", as the chronicles note,"),
        ("{c}, scholars agree.", ", as scholars agree,"),
    ] {
        assert!(echoes(frame, Some(aside)), "{frame} / {aside}");
    }
    assert!(!echoes(
        "Records show that {c}.",
        Some(", by all accounts,")
    ));
    for pred in Pred::ALL {
        let s = sentence_phrasings(pred);
        for form in s.seen.iter().chain(&s.val_held).chain(&s.held_out) {
            let lower = form.to_ascii_lowercase();
            for word in [
                "accounts",
                "elders",
                "chronicles",
                "scholars",
                "agree",
                "note",
            ] {
                assert!(lower.matches(word).count() <= 1, "{pred:?}: {form}");
            }
        }
    }
}
