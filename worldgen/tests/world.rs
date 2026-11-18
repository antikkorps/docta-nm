//! World generation invariants, checked on many seeds.

use docta_worldgen::model::{Kind, Obj, Pred, World};
use docta_worldgen::world::generate;
use std::collections::BTreeSet;

fn worlds() -> Vec<World> {
    (0..200)
        .map(|i| generate(&format!("w_{i:06}"), i))
        .collect()
}

fn int(w: &World, subj: &str, pred: Pred) -> i64 {
    match w.object(subj, pred) {
        Some(Obj::Int(n)) => *n,
        other => panic!("{subj} {pred:?}: expected int, got {other:?}"),
    }
}

fn target<'a>(w: &'a World, subj: &str, pred: Pred, kind: Kind) -> &'a str {
    match w.object(subj, pred) {
        Some(Obj::Entity(id)) if w.entity(id).is_some_and(|e| e.kind == kind) => id,
        other => panic!("{subj} {pred:?}: expected a {kind:?}, got {other:?}"),
    }
}

/// FNV-1a: a dependency-free, stable hash for golden values.
fn fnv1a(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[test]
fn is_deterministic() {
    assert_eq!(generate("w_1", 9), generate("w_1", 9));
    assert_ne!(generate("w_1", 9), generate("w_1", 10));
}

#[test]
fn output_is_pinned() {
    // Fails when sampling changes (rand upgrade, reordered draws): regenerate the
    // datasets on purpose, then update this value.
    let dump: String = (0..20).map(|i| format!("{:?}", generate("w", i))).collect();
    assert_eq!(fnv1a(&dump), 9646573538948596643);
}

#[test]
fn numeric_values_never_tie_within_a_kind() {
    // A tie would make a `compare` question unanswerable.
    for w in worlds() {
        for pred in [Pred::FoundedIn, Pred::Population, Pred::BornIn] {
            for kind in [Kind::City, Kind::Guild, Kind::Person] {
                let values: Vec<i64> = w
                    .entities
                    .iter()
                    .filter(|e| e.kind == kind && kind.preds().contains(&pred))
                    .map(|e| int(&w, &e.id, pred))
                    .collect();
                let distinct: BTreeSet<_> = values.iter().collect();
                assert_eq!(distinct.len(), values.len(), "{kind:?} {pred:?}");
            }
        }
    }
}

#[test]
fn ids_are_unique() {
    for w in worlds() {
        let e: BTreeSet<_> = w.entities.iter().map(|e| &e.id).collect();
        let f: BTreeSet<_> = w.facts.iter().map(|f| &f.id).collect();
        assert_eq!(e.len(), w.entities.len());
        assert_eq!(f.len(), w.facts.len());
    }
}

#[test]
fn entity_counts_are_in_range() {
    for w in worlds() {
        let count = |k| w.entities.iter().filter(|e| e.kind == k).count();
        assert!((3..=5).contains(&count(Kind::City)));
        assert!((2..=3).contains(&count(Kind::Guild)));
        assert!((4..=8).contains(&count(Kind::Person)));
    }
}

#[test]
fn each_entity_has_exactly_its_predicates() {
    for w in worlds() {
        for e in &w.entities {
            let preds: Vec<_> = w
                .facts
                .iter()
                .filter(|f| f.subj == e.id)
                .map(|f| f.pred)
                .collect();
            assert_eq!(preds, e.kind.preds(), "{}", e.id);
        }
    }
}

#[test]
fn objects_match_the_predicate_target() {
    for w in worlds() {
        for f in &w.facts {
            let actual = match &f.obj {
                Obj::Int(_) => None,
                Obj::Entity(id) => Some(w.entity(id).expect("dangling reference").kind),
            };
            assert_eq!(actual, f.pred.target(), "{}", f.id);
        }
    }
}

#[test]
fn predicate_tables_are_the_agreed_ones() {
    assert_eq!(Kind::City.preds(), [Pred::FoundedIn, Pred::Population]);
    assert_eq!(Kind::Guild.preds(), [Pred::FoundedIn, Pred::BasedIn]);
    assert_eq!(
        Kind::Person.preds(),
        [Pred::BornIn, Pred::LivesIn, Pred::MemberOf]
    );
    assert_eq!(Pred::LivesIn.target(), Some(Kind::City));
    assert_eq!(Pred::MemberOf.target(), Some(Kind::Guild));
    assert_eq!(Pred::Population.target(), None);
}

#[test]
fn references_point_to_the_right_kind_and_dates_are_coherent() {
    for w in worlds() {
        for e in w.entities.iter().filter(|e| e.kind == Kind::Guild) {
            let city = target(&w, &e.id, Pred::BasedIn, Kind::City);
            assert!(int(&w, &e.id, Pred::FoundedIn) > int(&w, city, Pred::FoundedIn));
        }
        for e in w.entities.iter().filter(|e| e.kind == Kind::Person) {
            let city = target(&w, &e.id, Pred::LivesIn, Kind::City);
            let guild = target(&w, &e.id, Pred::MemberOf, Kind::Guild);
            let born = int(&w, &e.id, Pred::BornIn);
            assert!(born > int(&w, city, Pred::FoundedIn));
            assert!(born > int(&w, guild, Pred::FoundedIn));
        }
    }
}
