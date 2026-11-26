//! Procedural generator of fictional worlds, their sources and questions.
//! Deterministic by construction: no LLM, no OS randomness.

pub mod clauses;
pub mod corpus;
pub mod dataset;
pub mod export;
pub mod frames;
pub mod grammar;
pub mod jsonl;
pub mod liars;
pub mod model;
pub mod names;
pub mod phrasings;
pub mod pronouns;
pub mod questions;
pub mod seed;
pub mod sources;
pub mod split;
pub mod templates;
pub mod traps;
pub mod world;
