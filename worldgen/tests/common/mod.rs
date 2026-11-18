//! Helpers shared by the corpus-level integration tests.

#![allow(dead_code)] // each test crate uses a different subset

use docta_worldgen::corpus::{Corpus, Options, Source, build};
use docta_worldgen::model::{Fact, Obj};
use docta_worldgen::split::Split;
use docta_worldgen::world::generate;

pub const DEFAULT: Options = Options {
    unanswerable_ratio: 0.33,
    max_hops: 2,
};

/// The 200 worlds most tests run on.
pub fn corpora(opts: &Options) -> Vec<Corpus> {
    corpora_n(200, opts)
}

pub fn fact<'a>(c: &'a Corpus, id: &str) -> &'a Fact {
    c.world
        .facts
        .iter()
        .find(|f| f.id == id)
        .expect("unknown fact")
}

pub fn name(c: &Corpus, id: &str) -> String {
    c.world.entity(id).expect("unknown entity").name.clone()
}

pub fn render(c: &Corpus, obj: &Obj) -> String {
    match obj {
        Obj::Int(n) => n.to_string(),
        Obj::Entity(id) => name(c, id),
    }
}

pub fn sources_with<'a>(c: &'a Corpus, fact_id: &str) -> Vec<&'a Source> {
    c.sources
        .iter()
        .filter(|s| s.facts.iter().any(|f| f == fact_id))
        .collect()
}

/// Worlds 0..n built with `opts`, as the train split writes them.
pub fn corpora_n(n: u64, opts: &Options) -> Vec<Corpus> {
    corpora_in(Split::Train, n, opts)
}

/// Worlds 0..n built with `opts` as `split` writes them.
pub fn corpora_in(split: Split, n: u64, opts: &Options) -> Vec<Corpus> {
    (0..n)
        .map(|i| build(generate(&format!("w_{i:06}"), i), split, opts))
        .collect()
}
