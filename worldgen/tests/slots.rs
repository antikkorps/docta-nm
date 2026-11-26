//! A world must get all its questions: a slot left empty would shift the
//! "9 questions, 3 unanswerable" mix every score assumes, without a word.

use docta_worldgen::corpus::Options;
use docta_worldgen::model::{Entity, Fact, Kind, Obj, Pred, World};
use docta_worldgen::questions::generate;
use docta_worldgen::sources::{withhold, write};
use docta_worldgen::split::Wording;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

/// One city and its three facts: far too few for nine distinct questions.
fn tiny_world() -> World {
    let fact = |id: &str, pred, n| Fact {
        id: id.into(),
        subj: "e1".into(),
        pred,
        obj: Obj::Int(n),
    };
    World {
        world_id: "w_tiny".into(),
        seed: 1,
        entities: vec![Entity {
            id: "e1".into(),
            kind: Kind::City,
            name: "Brask".into(),
        }],
        facts: vec![
            fact("f1", Pred::FoundedIn, 1820),
            fact("f2", Pred::Population, 900),
            fact("f3", Pred::Elevation, 210),
        ],
    }
}

#[test]
#[should_panic(expected = "w_tiny: no question for slot")]
fn a_slot_no_draw_can_fill_stops_generation() {
    let world = tiny_world();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let hidden = withhold(&world, &mut rng);
    let sources = write(&world, &hidden, Wording::Seen, &mut rng);
    let opts = Options {
        unanswerable_ratio: 0.33,
        max_hops: 2,
        trap_ratio: 0.0,
    };
    generate(&world, &sources, &hidden, 3, &opts, Wording::Seen, &mut rng);
}
