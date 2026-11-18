//! Procedural generator of fictional worlds, their sources and questions.
//! Deterministic by construction: no LLM, no OS randomness.

pub mod corpus;
pub mod dataset;
pub mod jsonl;
pub mod model;
pub mod names;
pub mod questions;
pub mod seed;
pub mod sources;
pub mod split;
pub mod templates;
pub mod world;
