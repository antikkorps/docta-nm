//! Source documents: the world's facts, minus the withheld ones, spread over
//! short template-written texts.

use crate::corpus::{Phrasing, Source};
use crate::model::{Fact, Obj, World};
use crate::pronouns::Pronoun;
use crate::split::Wording;
use crate::templates;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, RngExt};
use std::collections::BTreeSet;

/// Picks one fact per entity that no source will state. Every entity is thus
/// equally under-documented, so how often a subject is cited never tells
/// whether a question about it is answerable.
pub fn withhold<R: Rng>(world: &World, rng: &mut R) -> BTreeSet<String> {
    world
        .entities
        .iter()
        .map(|e| {
            let own: Vec<&Fact> = world.facts.iter().filter(|f| f.subj == e.id).collect();
            own.choose(rng).expect("every entity has facts").id.clone()
        })
        .collect()
}

/// Odds that the next fact of a source is about the subject just named, when
/// one is left: a pronoun needs its subject named just before.
const SAME_SUBJECT: f64 = 0.3;
/// Odds that a later sentence about a subject already named says its pronoun:
/// high, since grouping by subject is kept low (more of it would cite each
/// subject in fewer sources and widen the "many mentions => trap" gap).
const BY_PRONOUN: f64 = 0.7;

/// Spreads every non-withheld fact over sources of 1 to 3 facts each. Shuffling
/// first means a 2-hop question often needs two different sources.
pub fn write<R: Rng>(
    world: &World,
    withheld: &BTreeSet<String>,
    wording: Wording,
    rng: &mut R,
) -> Vec<Source> {
    let mut visible: Vec<&Fact> = world
        .facts
        .iter()
        .filter(|f| !withheld.contains(&f.id))
        .collect();
    visible.shuffle(rng);
    let mut sources = Vec::new();
    while !visible.is_empty() {
        let size = rng.random_range(1..=3).min(visible.len());
        let group = take_group(&mut visible, size, rng);
        sources.push(source(world, sources.len() + 1, &group, wording, rng));
    }
    sources
}

/// The next `size` facts, each one at `SAME_SUBJECT` odds about the subject of
/// the fact before it when such a fact is left: a pronoun needs that order.
fn take_group<'a, R: Rng>(rest: &mut Vec<&'a Fact>, size: usize, rng: &mut R) -> Vec<&'a Fact> {
    let mut group = vec![rest.remove(0)];
    while group.len() < size {
        let last = &group[group.len() - 1].subj;
        let same = rest.iter().position(|f| &f.subj == last);
        let at = match same {
            Some(i) if rng.random_bool(SAME_SUBJECT) => i,
            _ => 0,
        };
        group.push(rest.remove(at));
    }
    group
}

fn source<R: Rng>(
    world: &World,
    n: usize,
    facts: &[&Fact],
    wording: Wording,
    rng: &mut R,
) -> Source {
    let written: Vec<templates::Written> = facts
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let named = templates::sentence(
                f.pred,
                name(world, &f.subj),
                &render(world, &f.obj),
                wording,
                rng,
            );
            // Only right after a sentence about the same subject: with another
            // name in between, even a person could not tell whom it means.
            let again = i > 0 && facts[i - 1].subj == f.subj;
            match again && rng.random_bool(BY_PRONOUN) {
                true => pronoun_of(world, f, &named).unwrap_or(named),
                false => named,
            }
        })
        .collect();
    let text: Vec<&str> = written.iter().map(|w| w.text.as_str()).collect();
    let phrasings = facts
        .iter()
        .zip(&written)
        .map(|(f, w)| Phrasing {
            fact: f.id.clone(),
            template: w.template.clone(),
        })
        .collect();
    Source {
        id: format!("s{n}"),
        title: templates::title(name(world, &facts[0].subj), rng),
        // Hundredths keep the JSON value short and exact ("0.85").
        reliability: f64::from(rng.random_range(60..=100_u8)) / 100.0,
        text: text.join(" "),
        phrasings,
        facts: facts.iter().map(|f| f.id.clone()).collect(),
        false_claims: Vec::new(),
        copy_of: None,
    }
}

/// `named` with its subject said by pronoun, where one reads naturally.
fn pronoun_of(
    world: &World,
    fact: &Fact,
    named: &templates::Written,
) -> Option<templates::Written> {
    let subject = world
        .entity(&fact.subj)
        .expect("facts reference existing entities");
    templates::by_pronoun(
        named,
        Pronoun::of(world.seed, subject),
        &render(world, &fact.obj),
    )
}

/// Name of an entity of the world.
pub fn name<'a>(world: &'a World, id: &str) -> &'a str {
    &world
        .entity(id)
        .expect("facts reference existing entities")
        .name
}

/// How an object reads in text and in answers.
pub fn render(world: &World, obj: &Obj) -> String {
    match obj {
        Obj::Int(n) => n.to_string(),
        Obj::Entity(id) => name(world, id).to_string(),
    }
}
