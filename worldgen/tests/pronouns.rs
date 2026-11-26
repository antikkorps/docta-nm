//! A pronoun always follows a sentence of its own source that names its
//! subject, and the sources traps add never use one: a copy or a false claim
//! read alone must still say whom it is about.

mod common;

use common::{DEFAULT, corpora};
use docta_worldgen::corpus::{Corpus, Source};

/// Phrasings of a source whose template no longer names its subject.
fn pronoun_facts(s: &Source) -> Vec<usize> {
    (0..s.phrasings.len())
        .filter(|&i| !s.phrasings[i].template.contains("{s}"))
        .collect()
}

fn subject<'a>(c: &'a Corpus, fact: &str) -> &'a str {
    &c.world.fact(fact).expect("a stated fact").subj
}

#[test]
fn many_sources_name_a_subject_again_by_pronoun() {
    let all: Vec<Corpus> = corpora(&DEFAULT);
    let long = all
        .iter()
        .flat_map(|c| &c.sources)
        .filter(|s| s.phrasings.len() > 1)
        .count();
    let with = all
        .iter()
        .flat_map(|c| &c.sources)
        .filter(|s| !pronoun_facts(s).is_empty())
        .count();
    assert!(
        with * 5 >= long,
        "{with} of {long} sources of two facts or more"
    );
}

#[test]
fn a_pronoun_follows_a_sentence_about_the_same_subject() {
    // The sentence just before: with another name in between, "she" could
    // mean either. A source never opens on a pronoun.
    for c in corpora(&DEFAULT) {
        for s in &c.sources {
            for i in pronoun_facts(s) {
                let who = subject(&c, &s.phrasings[i].fact);
                assert!(
                    i > 0 && subject(&c, &s.phrasings[i - 1].fact) == who,
                    "{}: {}",
                    c.world.world_id,
                    s.text
                );
            }
        }
    }
}

#[test]
fn trap_sources_name_their_subject() {
    for c in corpora(&DEFAULT) {
        for s in c
            .sources
            .iter()
            .filter(|s| !s.false_claims.is_empty() || s.copy_of.is_some())
        {
            assert!(pronoun_facts(s).is_empty(), "{}", s.text);
        }
    }
}
