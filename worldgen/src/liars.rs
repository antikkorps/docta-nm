//! The sources a trap adds: a false claim about one fact, written like any
//! other source, and copies that repeat it while crediting their origin.

use crate::corpus::Source;
use crate::model::{Fact, Obj, World};
use crate::sources::{name, render};
use crate::templates;
use rand::seq::IndexedRandom;
use rand::{Rng, RngExt};
use std::collections::BTreeSet;

/// How a copy credits its origin. The held-out one is for ood only.
pub const ATTRIBUTIONS: [&str; 2] = ["According to {t}, {x}", "As {t} reports, {x}"];
pub const HELD_OUT_ATTRIBUTION: &str = "{t} records that {x}";

/// One wrong value for one fact, and how to write sources stating it.
pub struct Claim<'a> {
    pub world: &'a World,
    pub fact: &'a Fact,
    pub wrong: Obj,
    /// Prefix of temporary ids, unique within the world until renumbering.
    pub tag: String,
    pub held_out: bool,
    pub titles: &'a mut Titles,
}

/// The titles a world's sources use. A cited origin's title must name it
/// alone, or a copy's attribution would point at several sources.
pub struct Titles {
    used: BTreeSet<String>,
    cited: BTreeSet<String>,
}

impl Titles {
    pub fn of(sources: &[Source]) -> Self {
        let used = sources.iter().map(|s| s.title.clone()).collect();
        Self {
            used,
            cited: BTreeSet::new(),
        }
    }

    /// A title no source uses yet, reserved for an origin a copy will cite.
    fn for_origin<R: Rng>(&mut self, subj: &str, rng: &mut R) -> String {
        let free = self.pick(subj, |t| !self.used.contains(t), rng);
        let title = free.expect("eight title forms outnumber one subject's sources");
        self.used.insert(title.clone());
        self.cited.insert(title.clone());
        title
    }

    /// Any title but a cited one: copies may share titles, origins may not.
    fn for_copy<R: Rng>(&mut self, subj: &str, rng: &mut R) -> String {
        let title = self.pick(subj, |t| !self.cited.contains(t), rng);
        let title = title.expect("cited titles never fill every form");
        self.used.insert(title.clone());
        title
    }

    fn pick<R: Rng>(
        &self,
        subj: &str,
        ok: impl Fn(&String) -> bool,
        rng: &mut R,
    ) -> Option<String> {
        let candidates: Vec<String> = templates::titles(subj).into_iter().filter(ok).collect();
        candidates.choose(rng).cloned()
    }
}

impl<'a> Claim<'a> {
    pub fn new<R: Rng>(
        world: &'a World,
        fact: &'a Fact,
        tag: String,
        held_out: bool,
        titles: &'a mut Titles,
        rng: &mut R,
    ) -> Self {
        let wrong = wrong_value(world, fact, rng);
        Self {
            world,
            fact,
            wrong,
            tag,
            held_out,
            titles,
        }
    }

    /// An independent source stating the wrong value, under a title no other
    /// source of the world uses, so a copy citing it names it alone.
    pub fn origin<R: Rng>(&mut self, reliability: f64, rng: &mut R) -> Source {
        let subj = name(self.world, &self.fact.subj);
        let wrong = render(self.world, &self.wrong);
        Source {
            id: format!("{}:origin", self.tag),
            title: self.titles.for_origin(subj, rng),
            reliability,
            text: templates::sentence(self.fact.pred, subj, &wrong, self.held_out, rng),
            facts: Vec::new(),
            false_claims: vec![(self.fact.id.clone(), self.wrong.clone())],
            copy_of: None,
        }
    }

    /// A copy repeats its origin's sentence and says where it took it from.
    /// Its mid reliability keeps repetition from looking like good sourcing.
    pub fn copy<R: Rng>(&mut self, origin: &Source, k: u8, rng: &mut R) -> Source {
        let phrasing = if self.held_out {
            HELD_OUT_ATTRIBUTION
        } else {
            ATTRIBUTIONS.choose(rng).expect("non-empty")
        };
        let claim = crate::grammar::lower_opening(&origin.text);
        Source {
            id: format!("{}:copy{k}", origin.id),
            title: self.titles.for_copy(name(self.world, &self.fact.subj), rng),
            reliability: cents(rng, 60, 85),
            text: phrasing
                .replace("{t}", &origin.title)
                .replace("{x}", &claim),
            facts: Vec::new(),
            false_claims: origin.false_claims.clone(),
            copy_of: Some(origin.id.clone()),
        }
    }
}

/// Another value of the same kind: a nearby number, or another entity.
fn wrong_value<R: Rng>(world: &World, fact: &Fact, rng: &mut R) -> Obj {
    match &fact.obj {
        Obj::Int(n) => {
            let delta = rng.random_range(1..=(n / 10).max(5));
            let up = rng.random_bool(0.5) || *n - delta <= 0;
            Obj::Int(if up { n + delta } else { n - delta })
        }
        Obj::Entity(id) => {
            let kind = fact
                .pred
                .target()
                .expect("entity objects have a target kind");
            let others: Vec<_> = world
                .entities
                .iter()
                .filter(|e| e.kind == kind && &e.id != id)
                .collect();
            let other = others.choose(rng).expect("two entities per kind");
            Obj::Entity(other.id.clone())
        }
    }
}

/// A reliability in hundredths, like ordinary sources.
pub fn cents<R: Rng>(rng: &mut R, low: u8, high: u8) -> f64 {
    f64::from(rng.random_range(low..=high.max(low))) / 100.0
}
