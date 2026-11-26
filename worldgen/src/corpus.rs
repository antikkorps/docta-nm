//! What a model gets to read (sources) and is asked (questions) about a world.

use crate::model::Obj;
use crate::model::World;
use crate::split::Split;
use crate::templates::Written;
use crate::{questions, sources, traps};
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
    /// Ceiling on the share of answerable retrieve questions whose fact gets
    /// a trap; the real share is lower (crate::dataset::Counts).
    pub trap_ratio: f64,
}

/// Disagreeing sources around the fact a question asks about (since JSONL v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trap {
    /// One independent false source, clearly less reliable than the true one.
    Contradiction,
    /// One false origin repeated by attributed copies, against a reliable primary.
    FalseConsensus,
    /// Two independent, equally reliable sources state opposite values.
    Undecidable,
}

impl Trap {
    pub fn as_str(self) -> &'static str {
        match self {
            Trap::Contradiction => "contradiction",
            Trap::FalseConsensus => "false_consensus",
            Trap::Undecidable => "undecidable",
        }
    }
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

/// The template a sentence of a source was written from, and the fact it
/// states. Ground truth for diagnosis (which wording a core fails on), never
/// exposed to a model.
#[derive(Debug, Clone, PartialEq)]
pub struct Phrasing {
    pub fact: String,
    pub template: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub id: String,
    pub title: String,
    pub reliability: f64,
    pub text: String,
    /// Ground truth: one per stated sentence; a copy, which wraps its origin's
    /// text in an attribution, carries none.
    pub phrasings: Vec<Phrasing>,
    /// Ground truth, never exposed to a model.
    pub facts: Vec<String>,
    /// Ground truth: facts this source states with a wrong object.
    pub false_claims: Vec<(String, Obj)>,
    /// Ground truth: the source whose claim this one repeats, if a copy.
    pub copy_of: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub id: String,
    pub text: String,
    /// Ground truth: the template that wrote `text`, names as `{s}` (`{o}`).
    pub template: String,
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
    /// Every world fact the question rests on, stated or not. Internal, used to
    /// test the ood seal on unanswerable questions too; not serialised.
    pub support: Vec<String>,
    pub trap: Option<Trap>,
    /// The emergency exit the method should take, if any.
    pub expected_exit: Option<&'static str>,
}

impl Question {
    pub fn answerable(&self) -> bool {
        self.answer.is_some()
    }

    pub(crate) fn answerable_with(
        written: Written,
        shape: Primitive,
        hops: u8,
        answer: String,
        required_facts: Vec<String>,
        required_sources: Vec<String>,
    ) -> Self {
        Self {
            support: required_facts.clone(),
            id: String::new(),
            text: written.text,
            template: written.template,
            primitive: shape,
            shape,
            hops,
            answer: Some(answer),
            missing: None,
            required_facts,
            required_sources,
            absent_fact: None,
            trap: None,
            expected_exit: None,
        }
    }

    pub(crate) fn unanswerable(
        written: Written,
        shape: Primitive,
        hops: u8,
        missing: String,
        absent_fact: String,
        support: Vec<String>,
    ) -> Self {
        Self {
            support,
            id: String::new(),
            text: written.text,
            template: written.template,
            primitive: Primitive::SpotMissing,
            shape,
            hops,
            answer: None,
            missing: Some(missing),
            required_facts: Vec::new(),
            required_sources: Vec::new(),
            absent_fact: Some(absent_fact),
            trap: None,
            expected_exit: None,
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
pub fn build(world: World, split: Split, opts: &Options) -> Corpus {
    let mut rng = ChaCha8Rng::seed_from_u64(world.seed);
    rng.set_stream(1);
    let n_missing = unanswerable_count(opts.unanswerable_ratio);
    let withheld = sources::withhold(&world, &mut rng);
    let wording = split.wording();
    let mut srcs = sources::write(&world, &withheld, wording, &mut rng);
    let mut questions =
        questions::generate(&world, &srcs, &withheld, n_missing, opts, wording, &mut rng);
    // Drawn last: without traps, every earlier draw and byte stays as it was.
    let ground = traps::Ground {
        world: &world,
        withheld: &withheld,
        wording,
    };
    traps::apply(
        &ground,
        &mut srcs,
        &mut questions,
        opts.trap_ratio,
        &mut rng,
    );
    Corpus {
        world,
        sources: srcs,
        questions,
    }
}

fn unanswerable_count(ratio: f64) -> usize {
    (QUESTIONS_PER_WORLD as f64 * ratio).round() as usize
}
