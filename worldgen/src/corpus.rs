//! What a model gets to read (sources) and is asked (questions) about a world.

use crate::model::World;
use crate::{questions, sources};
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

/// Questions per world; `round(QUESTIONS_PER_WORLD * unanswerable_ratio)` of
/// them ask about a fact no source states. Every world has at least as many
/// entities, each with one unstated fact, so there are always enough.
pub const QUESTIONS_PER_WORLD: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub unanswerable_ratio: f64,
    pub max_hops: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primitive {
    Retrieve,
    Chain,
    Compare,
    SpotMissing,
}

impl Primitive {
    pub fn as_str(self) -> &'static str {
        match self {
            Primitive::Retrieve => "retrieve",
            Primitive::Chain => "chain",
            Primitive::Compare => "compare",
            Primitive::SpotMissing => "spot_missing",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub id: String,
    pub title: String,
    pub reliability: f64,
    pub text: String,
    /// Ground truth, never exposed to a model.
    pub facts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub text: String,
    /// `SpotMissing` for every unanswerable question, whatever its shape.
    pub primitive: Primitive,
    /// Retrieve, Chain or Compare. Internal: unanswerable questions take every
    /// shape, so the wording never gives abstention away.
    pub shape: Primitive,
    pub hops: u8,
    pub answer: Option<String>,
    pub missing: Option<String>,
    pub required_facts: Vec<String>,
    pub required_sources: Vec<String>,
    /// For an unanswerable question: the world fact it asks about, which no
    /// source mentions. Internal, used to test the guarantee; not serialised.
    pub absent_fact: Option<String>,
}

impl Question {
    pub fn answerable(&self) -> bool {
        self.answer.is_some()
    }

    pub(crate) fn answerable_with(
        text: String,
        shape: Primitive,
        hops: u8,
        answer: String,
        required_facts: Vec<String>,
        required_sources: Vec<String>,
    ) -> Self {
        Self {
            id: String::new(),
            text,
            primitive: shape,
            shape,
            hops,
            answer: Some(answer),
            missing: None,
            required_facts,
            required_sources,
            absent_fact: None,
        }
    }

    pub(crate) fn unanswerable(
        text: String,
        shape: Primitive,
        hops: u8,
        missing: String,
        absent_fact: String,
    ) -> Self {
        Self {
            id: String::new(),
            text,
            primitive: Primitive::SpotMissing,
            shape,
            hops,
            answer: None,
            missing: Some(missing),
            required_facts: Vec::new(),
            required_sources: Vec::new(),
            absent_fact: Some(absent_fact),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Corpus {
    pub world: World,
    pub sources: Vec<Source>,
    pub questions: Vec<Question>,
}

/// Writes the sources and questions of `world`. Uses its own RNG stream, so
/// changing a template never changes the worlds themselves.
pub fn build(world: World, opts: &Options) -> Corpus {
    let mut rng = ChaCha8Rng::seed_from_u64(world.seed);
    rng.set_stream(1);
    let n_missing = unanswerable_count(opts.unanswerable_ratio);
    let withheld = sources::withhold(&world, &mut rng);
    let srcs = sources::write(&world, &withheld, &mut rng);
    let questions = questions::generate(&world, &srcs, &withheld, n_missing, opts, &mut rng);
    Corpus {
        world,
        sources: srcs,
        questions,
    }
}

fn unanswerable_count(ratio: f64) -> usize {
    (QUESTIONS_PER_WORLD as f64 * ratio).round() as usize
}
