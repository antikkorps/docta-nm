//! The seen phrasings as one JSON file, for the orchestrator's rules.
//!
//! The rules must know exactly what a core trained on train sees, never the
//! phrasings held out for val or ood. A hand copy could not follow the
//! combinatorial grammar, so worldgen writes the file and a test keeps the committed copy
//! (orchestrator/src/docta_orchestrator/models/templates_seen.json) in step.

use crate::grammar::{sentence_phrasings, with_pronouns};
use crate::model::Pred;
use crate::templates::{comparison_phrasings, link_phrasings, question_phrasings};
use serde_json::{Map, Value, json};

/// Every seen phrasing, keyed by kind then predicate; keys sort, so the
/// output is byte-stable.
pub fn seen_templates_json() -> String {
    let mut sentences = Map::new();
    let mut questions = Map::new();
    let mut comparisons = Map::new();
    let mut links = Map::new();
    for pred in Pred::ALL {
        let key = pred.as_str().to_string();
        sentences.insert(
            key.clone(),
            json!(with_pronouns(pred, sentence_phrasings(pred).seen)),
        );
        questions.insert(key.clone(), json!(question_phrasings(pred).seen));
        if let Some(c) = comparison_phrasings(pred) {
            comparisons.insert(key.clone(), json!(c.seen[0]));
        }
        if let Some(l) = link_phrasings(pred) {
            links.insert(key, json!(l.seen[0]));
        }
    }
    let all = json!({
        "sentences": Value::Object(sentences),
        "questions": Value::Object(questions),
        "comparisons": Value::Object(comparisons),
        "chain_links": Value::Object(links),
        // Copies and titles write these words too: never names either.
        "attributions": json!(crate::liars::ATTRIBUTIONS),
        "titles": json!(crate::templates::titles("{s}")),
    });
    serde_json::to_string_pretty(&all).expect("plain strings serialise") + "\n"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    const COMMITTED: &str = "../orchestrator/src/docta_orchestrator/models/templates_seen.json";

    #[test]
    fn the_committed_copy_is_what_worldgen_writes() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(COMMITTED);
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            committed == seen_templates_json(),
            "run `just templates` to refresh {COMMITTED}"
        );
    }

    #[test]
    fn no_held_out_phrasing_is_exported() {
        let json = seen_templates_json();
        for pred in crate::model::Pred::ALL {
            for held in crate::grammar::sentence_phrasings(pred).held_out {
                assert!(!json.contains(&format!("\"{held}\"")), "{held}");
            }
            assert!(!json.contains(crate::templates::question_phrasings(pred).held_out));
        }
    }
}
