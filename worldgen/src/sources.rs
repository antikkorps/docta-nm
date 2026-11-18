//! Source documents: the world's facts, minus the withheld ones, spread over
//! short template-written texts.

use crate::corpus::Source;
use crate::model::{Fact, Obj, World};
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

/// Spreads every non-withheld fact over sources of 1 to 3 facts each. Shuffling
/// first means a 2-hop question often needs two different sources.
pub fn write<R: Rng>(
    world: &World,
    withheld: &BTreeSet<String>,
    held_out: bool,
    rng: &mut R,
) -> Vec<Source> {
    let mut visible: Vec<&Fact> = world
        .facts
        .iter()
        .filter(|f| !withheld.contains(&f.id))
        .collect();
    visible.shuffle(rng);
    let mut sources = Vec::new();
    let mut rest = visible.as_slice();
    while !rest.is_empty() {
        let size = rng.random_range(1..=3).min(rest.len());
        let (group, tail) = rest.split_at(size);
        sources.push(source(world, sources.len() + 1, group, held_out, rng));
        rest = tail;
    }
    sources
}

fn source<R: Rng>(world: &World, n: usize, facts: &[&Fact], held_out: bool, rng: &mut R) -> Source {
    let text: Vec<String> = facts
        .iter()
        .map(|f| {
            templates::sentence(
                f.pred,
                name(world, &f.subj),
                &render(world, &f.obj),
                held_out,
                rng,
            )
        })
        .collect();
    Source {
        id: format!("s{n}"),
        title: templates::title(name(world, &facts[0].subj), rng),
        // Hundredths keep the JSON value short and exact ("0.85").
        reliability: f64::from(rng.random_range(60..=100_u8)) / 100.0,
        text: text.join(" "),
        facts: facts.iter().map(|f| f.id.clone()).collect(),
        false_claims: Vec::new(),
        copy_of: None,
    }
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
