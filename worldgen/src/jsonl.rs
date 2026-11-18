//! JSONL v1 serialisation: one line per world, source and question.
//! Field order is fixed by the record structs, so output is byte-stable.

use crate::corpus::{Corpus, Question, Source};
use crate::model::{Entity, Fact, Obj};
use crate::split::Split;
use serde::Serialize;

pub const SCHEMA: &str = "v1";

#[derive(Serialize)]
struct WorldRecord<'a> {
    schema: &'static str,
    r#type: &'static str,
    world_id: &'a str,
    seed: u64,
    split: &'static str,
    entities: Vec<EntityRecord<'a>>,
    facts: Vec<FactRecord<'a>>,
}

#[derive(Serialize)]
struct EntityRecord<'a> {
    id: &'a str,
    kind: &'static str,
    name: &'a str,
}

#[derive(Serialize)]
struct FactRecord<'a> {
    id: &'a str,
    subj: &'a str,
    pred: &'static str,
    obj: ObjValue<'a>,
}

/// A fact object is a bare number or an entity id string, as in the brief.
#[derive(Serialize)]
#[serde(untagged)]
enum ObjValue<'a> {
    Int(i64),
    Entity(&'a str),
}

#[derive(Serialize)]
struct SourceRecord<'a> {
    schema: &'static str,
    r#type: &'static str,
    world_id: &'a str,
    source_id: &'a str,
    title: &'a str,
    reliability: f64,
    text: &'a str,
    facts: &'a [String],
    false_claims: Vec<ClaimRecord<'a>>,
    copy_of: Option<&'a str>,
}

/// A fact a source states with a wrong object: ground truth, never exposed.
#[derive(Serialize)]
struct ClaimRecord<'a> {
    fact: &'a str,
    obj: ObjValue<'a>,
}

#[derive(Serialize)]
struct QuestionRecord<'a> {
    schema: &'static str,
    r#type: &'static str,
    world_id: &'a str,
    question_id: &'a str,
    text: &'a str,
    primitive: &'static str,
    hops: u8,
    answerable: bool,
    answer: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    missing: Option<&'a str>,
    required_facts: &'a [String],
    required_sources: &'a [String],
    trap: Option<&'static str>,
    expected_exit: Option<&'static str>,
}

/// All lines of one world: the world, then its sources, then its questions.
pub fn lines(corpus: &Corpus, split: Split) -> Vec<String> {
    let id = corpus.world.world_id.as_str();
    let mut out = vec![to_line(&world(corpus, split))];
    out.extend(corpus.sources.iter().map(|s| to_line(&source(id, s))));
    out.extend(corpus.questions.iter().map(|q| to_line(&question(id, q))));
    out
}

fn to_line<T: Serialize>(record: &T) -> String {
    serde_json::to_string(record).expect("records always serialise")
}

fn world(c: &Corpus, split: Split) -> WorldRecord<'_> {
    WorldRecord {
        schema: SCHEMA,
        r#type: "world",
        world_id: &c.world.world_id,
        seed: c.world.seed,
        split: split.as_str(),
        entities: c.world.entities.iter().map(entity).collect(),
        facts: c.world.facts.iter().map(fact).collect(),
    }
}

fn entity(e: &Entity) -> EntityRecord<'_> {
    EntityRecord {
        id: &e.id,
        kind: e.kind.as_str(),
        name: &e.name,
    }
}

fn fact(f: &Fact) -> FactRecord<'_> {
    FactRecord {
        id: &f.id,
        subj: &f.subj,
        pred: f.pred.as_str(),
        obj: value(&f.obj),
    }
}

fn value(obj: &Obj) -> ObjValue<'_> {
    match obj {
        Obj::Int(n) => ObjValue::Int(*n),
        Obj::Entity(id) => ObjValue::Entity(id),
    }
}

fn source<'a>(world_id: &'a str, s: &'a Source) -> SourceRecord<'a> {
    SourceRecord {
        schema: SCHEMA,
        r#type: "source",
        world_id,
        source_id: &s.id,
        title: &s.title,
        reliability: s.reliability,
        text: &s.text,
        facts: &s.facts,
        false_claims: s
            .false_claims
            .iter()
            .map(|(fact, obj)| ClaimRecord {
                fact,
                obj: value(obj),
            })
            .collect(),
        copy_of: s.copy_of.as_deref(),
    }
}

fn question<'a>(world_id: &'a str, q: &'a Question) -> QuestionRecord<'a> {
    QuestionRecord {
        schema: SCHEMA,
        r#type: "question",
        world_id,
        question_id: &q.id,
        text: &q.text,
        primitive: q.primitive.as_str(),
        hops: q.hops,
        answerable: q.answerable(),
        answer: q.answer.as_deref(),
        missing: q.missing.as_deref(),
        required_facts: &q.required_facts,
        required_sources: &q.required_sources,
        trap: q.trap.map(|t| t.as_str()),
        expected_exit: q.expected_exit,
    }
}
