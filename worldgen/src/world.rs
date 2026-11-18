//! Generation of one world's entities and facts from its seed.

use crate::model::{Entity, Fact, Kind, Obj, Pred, World};
use crate::names::NameGen;
use rand::rngs::ChaCha8Rng;
use rand::seq::IndexedRandom;
use rand::{RngExt, SeedableRng};
use std::collections::BTreeSet;
use std::ops::RangeInclusive;

// Disjoint year ranges make the chronology coherent by construction: cities
// predate guilds, which predate every living person.
const CITY_YEARS: RangeInclusive<i64> = 800..=1499;
const GUILD_YEARS: RangeInclusive<i64> = 1500..=1799;
const BIRTH_YEARS: RangeInclusive<i64> = 1820..=1999;
const POPULATION: RangeInclusive<i64> = 500..=500_000;

/// Builds the world `world_id` from `seed`; same inputs, same world.
pub fn generate(world_id: &str, seed: u64) -> World {
    let mut b = Builder::new(seed);
    let cities = b.cities();
    let guilds = b.guilds(&cities);
    b.persons(&cities, &guilds);
    World {
        world_id: world_id.to_string(),
        seed,
        entities: b.entities,
        facts: b.facts,
    }
}

struct Builder {
    rng: ChaCha8Rng,
    names: NameGen,
    entities: Vec<Entity>,
    facts: Vec<Fact>,
    drawn: BTreeSet<(Pred, i64)>,
}

impl Builder {
    fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            names: NameGen::default(),
            entities: Vec::new(),
            facts: Vec::new(),
            drawn: BTreeSet::new(),
        }
    }

    fn cities(&mut self) -> Vec<String> {
        let n = self.rng.random_range(3..=5);
        (0..n)
            .map(|_| {
                let name = self.names.token(&mut self.rng);
                let id = self.entity(Kind::City, name);
                self.int_fact(&id, Pred::FoundedIn, CITY_YEARS);
                self.int_fact(&id, Pred::Population, POPULATION);
                id
            })
            .collect()
    }

    fn guilds(&mut self, cities: &[String]) -> Vec<String> {
        let n = self.rng.random_range(2..=3);
        (0..n)
            .map(|_| {
                let name = self.names.token(&mut self.rng);
                let id = self.entity(Kind::Guild, name);
                self.int_fact(&id, Pred::FoundedIn, GUILD_YEARS);
                self.link_fact(&id, Pred::BasedIn, cities);
                id
            })
            .collect()
    }

    fn persons(&mut self, cities: &[String], guilds: &[String]) {
        let n = self.rng.random_range(4..=8);
        for _ in 0..n {
            let name = self.names.person(&mut self.rng);
            let id = self.entity(Kind::Person, name);
            self.int_fact(&id, Pred::BornIn, BIRTH_YEARS);
            self.link_fact(&id, Pred::LivesIn, cities);
            self.link_fact(&id, Pred::MemberOf, guilds);
        }
    }

    fn entity(&mut self, kind: Kind, name: String) -> String {
        let id = format!("e{}", self.entities.len() + 1);
        self.entities.push(Entity {
            id: id.clone(),
            kind,
            name,
        });
        id
    }

    /// Redraws until the value is new for this predicate: a tie would make a
    /// `compare` question unanswerable.
    fn int_fact(&mut self, subj: &str, pred: Pred, range: RangeInclusive<i64>) {
        let value = loop {
            let v = self.rng.random_range(range.clone());
            if self.drawn.insert((pred, v)) {
                break v;
            }
        };
        self.fact(subj, pred, Obj::Int(value));
    }

    fn link_fact(&mut self, subj: &str, pred: Pred, targets: &[String]) {
        let target = targets
            .choose(&mut self.rng)
            .expect("targets are non-empty");
        self.fact(subj, pred, Obj::Entity(target.clone()));
    }

    fn fact(&mut self, subj: &str, pred: Pred, obj: Obj) {
        let id = format!("f{}", self.facts.len() + 1);
        let subj = subj.to_string();
        self.facts.push(Fact {
            id,
            subj,
            pred,
            obj,
        });
    }
}
