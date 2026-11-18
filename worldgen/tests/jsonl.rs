//! The v0 line format, checked byte for byte against the brief.

use docta_worldgen::corpus::{Corpus, Primitive, Question, Source};
use docta_worldgen::jsonl::lines;
use docta_worldgen::model::{Entity, Fact, Kind, Obj, Pred, World};
use docta_worldgen::split::Split;

fn tiny() -> Corpus {
    let entity = |id: &str, kind, name: &str| Entity {
        id: id.into(),
        kind,
        name: name.into(),
    };
    let fact = |id: &str, subj: &str, pred, obj| Fact {
        id: id.into(),
        subj: subj.into(),
        pred,
        obj,
    };
    let question = |answer: Option<&str>, missing: Option<&str>| Question {
        id: "q1".into(),
        text: "In what year was Varnelle founded?".into(),
        shape: Primitive::Retrieve,
        primitive: if answer.is_some() {
            Primitive::Retrieve
        } else {
            Primitive::SpotMissing
        },
        hops: 1,
        answer: answer.map(Into::into),
        missing: missing.map(Into::into),
        required_facts: answer.map(|_| vec!["f1".into()]).unwrap_or_default(),
        required_sources: answer.map(|_| vec!["s1".into()]).unwrap_or_default(),
        absent_fact: None,
    };
    Corpus {
        world: World {
            world_id: "w_000123".into(),
            seed: 123,
            entities: vec![
                entity("e1", Kind::City, "Varnelle"),
                entity("e2", Kind::Person, "Orlan Teste"),
            ],
            facts: vec![
                fact("f1", "e1", Pred::FoundedIn, Obj::Int(1432)),
                fact("f2", "e2", Pred::LivesIn, Obj::Entity("e1".into())),
            ],
        },
        sources: vec![Source {
            id: "s1".into(),
            title: "Chronicle of Varnelle".into(),
            reliability: 0.9,
            text: "Varnelle was founded in 1432.".into(),
            facts: vec!["f1".into()],
        }],
        questions: vec![
            question(Some("1432"), None),
            question(None, Some("no source mentions it")),
        ],
    }
}

#[test]
fn matches_the_v0_format() {
    let out = lines(&tiny(), Split::Train);
    let expected = [
        r#"{"schema":"v0","type":"world","world_id":"w_000123","seed":123,"split":"train","entities":[{"id":"e1","kind":"city","name":"Varnelle"},{"id":"e2","kind":"person","name":"Orlan Teste"}],"facts":[{"id":"f1","subj":"e1","pred":"founded_in","obj":1432},{"id":"f2","subj":"e2","pred":"lives_in","obj":"e1"}]}"#,
        r#"{"schema":"v0","type":"source","world_id":"w_000123","source_id":"s1","title":"Chronicle of Varnelle","reliability":0.9,"text":"Varnelle was founded in 1432.","facts":["f1"]}"#,
        r#"{"schema":"v0","type":"question","world_id":"w_000123","question_id":"q1","text":"In what year was Varnelle founded?","primitive":"retrieve","hops":1,"answerable":true,"answer":"1432","required_facts":["f1"],"required_sources":["s1"]}"#,
        r#"{"schema":"v0","type":"question","world_id":"w_000123","question_id":"q1","text":"In what year was Varnelle founded?","primitive":"spot_missing","hops":1,"answerable":false,"answer":null,"missing":"no source mentions it","required_facts":[],"required_sources":[]}"#,
    ];
    assert_eq!(out, expected);
}
