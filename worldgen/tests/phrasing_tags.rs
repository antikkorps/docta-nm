//! Each sentence and question carries the template that wrote it: ground
//! truth for diagnosis (which wording a core fails on), never a model input.

mod common;

use common::{DEFAULT, corpora_in};
use docta_worldgen::corpus::{Corpus, Primitive};
use docta_worldgen::model::Obj;
use docta_worldgen::split::Split;

fn name(c: &Corpus, id: &str) -> String {
    c.world.entity(id).expect("an entity").name.clone()
}

fn object(c: &Corpus, obj: &Obj) -> String {
    match obj {
        Obj::Int(n) => n.to_string(),
        Obj::Entity(id) => name(c, id),
    }
}

#[test]
fn every_stated_sentence_carries_its_template() {
    for split in [Split::Train, Split::Val, Split::Test, Split::Ood] {
        for c in corpora_in(split, 40, &DEFAULT) {
            for s in c.sources.iter().filter(|s| s.copy_of.is_none()) {
                let stated = s.facts.len() + s.false_claims.len();
                assert_eq!(s.phrasings.len(), stated, "{}: {}", s.id, s.text);
                for p in &s.phrasings {
                    let fact = c.world.fact(&p.fact).expect("a world fact");
                    let wrong = s.false_claims.iter().find(|(f, _)| f == &p.fact);
                    let obj = wrong.map_or(&fact.obj, |(_, o)| o);
                    let filled = p
                        .template
                        .replace("{s}", &name(&c, &fact.subj))
                        .replace("{o}", &object(&c, obj));
                    assert!(s.text.contains(&filled), "{filled} not in {}", s.text);
                }
            }
        }
    }
}

#[test]
fn a_copy_carries_no_template_of_its_own() {
    for c in corpora_in(Split::Train, 40, &DEFAULT) {
        for s in c.sources.iter().filter(|s| s.copy_of.is_some()) {
            assert!(s.phrasings.is_empty(), "{}", s.text);
        }
    }
}

#[test]
fn every_question_carries_the_template_that_wrote_it() {
    for split in [Split::Train, Split::Val, Split::Test, Split::Ood] {
        for c in corpora_in(split, 40, &DEFAULT) {
            for q in &c.questions {
                let t = &q.template;
                assert!(t.contains("{s}"), "{t}");
                let names: Vec<String> = c.world.entities.iter().map(|e| e.name.clone()).collect();
                let fits = names.iter().any(|a| {
                    t.replace("{s}", a) == q.text
                        || (q.shape == Primitive::Compare
                            && names
                                .iter()
                                .any(|b| t.replace("{s}", a).replace("{o}", b) == q.text))
                });
                assert!(fits, "{t} does not write {}", q.text);
            }
        }
    }
}
