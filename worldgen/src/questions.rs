//! Questions over a world. Which slots lack an answer is decided first, then
//! each slot draws its shape (retrieve, chain, compare) independently, so the
//! question text alone never predicts abstention.

use crate::corpus::{Options, Primitive, QUESTIONS_PER_WORLD, Question, Source};
use crate::model::{Fact, Obj, Pred, World};
use crate::sources::{name, render};
use crate::templates;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, RngExt};
use std::collections::{BTreeMap, BTreeSet};

/// Chain combinations only the ood split may ask, so composing them there was
/// never seen in training.
pub const HELD_OUT_CHAINS: [(Pred, Pred); 2] = [
    (Pred::MemberOf, Pred::BasedIn),
    (Pred::BasedIn, Pred::Elevation),
];

// Redraws allowed per slot when a draw duplicates an earlier question.
const MAX_ATTEMPTS: usize = 1000;

/// A drawn question before numbering: its text, answer and the facts it rests on.
struct Draft<'a> {
    shape: Primitive,
    text: String,
    answer: String,
    facts: Vec<&'a Fact>,
}

pub fn generate<R: Rng>(
    world: &World,
    sources: &[Source],
    hidden: &BTreeSet<String>,
    n_missing: usize,
    opts: &Options,
    held_out: bool,
    rng: &mut R,
) -> Vec<Question> {
    let ctx = Ctx::new(world, sources, hidden, held_out);
    let mut shapes = vec![Primitive::Retrieve, Primitive::Compare];
    if opts.max_hops >= 2 {
        shapes.push(Primitive::Chain);
    }
    let mut answerable: Vec<bool> = (0..QUESTIONS_PER_WORLD).map(|i| i >= n_missing).collect();
    answerable.shuffle(rng);
    let mut asked = BTreeSet::new();
    let mut out = Vec::new();
    for want_answer in answerable {
        let found = (0..MAX_ATTEMPTS).find_map(|_| {
            let shape = *shapes.choose(rng).expect("non-empty");
            ctx.draw(shape, want_answer, rng)
                .filter(|d| asked.insert(key(d)))
        });
        out.extend(found.map(|d| ctx.finish(d, want_answer)));
    }
    for (i, q) in out.iter_mut().enumerate() {
        q.id = format!("q{}", i + 1);
    }
    out
}

/// Two drafts with one key ask the same thing, whatever their wording.
fn key(d: &Draft) -> (&'static str, Vec<String>) {
    let mut ids: Vec<String> = d.facts.iter().map(|f| f.id.clone()).collect();
    ids.sort();
    (d.shape.as_str(), ids)
}

struct Ctx<'a> {
    world: &'a World,
    hidden: &'a BTreeSet<String>,
    held_out: bool,
    source_of: BTreeMap<&'a str, &'a str>,
}

impl<'a> Ctx<'a> {
    fn new(
        world: &'a World,
        sources: &'a [Source],
        hidden: &'a BTreeSet<String>,
        held_out: bool,
    ) -> Self {
        let source_of = sources
            .iter()
            .flat_map(|s| s.facts.iter().map(move |f| (f.as_str(), s.id.as_str())))
            .collect();
        Self {
            world,
            hidden,
            held_out,
            source_of,
        }
    }

    fn stated(&self, f: &Fact) -> bool {
        !self.hidden.contains(&f.id)
    }

    /// A draft whose facts are all stated if `answerable`, else with at least
    /// one fact no source states.
    fn draw<R: Rng>(&self, shape: Primitive, answerable: bool, rng: &mut R) -> Option<Draft<'a>> {
        match shape {
            Primitive::Retrieve => self.retrieve(answerable, rng),
            Primitive::Chain => self.chain(answerable, rng),
            Primitive::Compare => self.compare(answerable, rng),
            Primitive::SpotMissing => None,
        }
    }

    /// Subject first, uniformly: every entity has the same chance whether the
    /// asked fact is stated or not.
    fn retrieve<R: Rng>(&self, answerable: bool, rng: &mut R) -> Option<Draft<'a>> {
        let subject = self.world.entities.choose(rng)?;
        let own: Vec<&Fact> = self
            .world
            .facts
            .iter()
            .filter(|f| f.subj == subject.id && self.stated(f) == answerable)
            .collect();
        let f = *own.choose(rng)?;
        let text = templates::question(f.pred, &subject.name, self.held_out);
        Some(Draft {
            shape: Primitive::Retrieve,
            text,
            answer: render(self.world, &f.obj),
            facts: vec![f],
        })
    }

    /// Two hops: a link fact, then a fact about the entity it points to.
    fn chain<R: Rng>(&self, answerable: bool, rng: &mut R) -> Option<Draft<'a>> {
        let (a, b) = self.pair(answerable, rng, |a, b| {
            // ood may ask any combination; the others never ask a held-out one.
            a.obj == Obj::Entity(b.subj.clone())
                && (self.held_out || !HELD_OUT_CHAINS.contains(&(a.pred, b.pred)))
        })?;
        let via = templates::describe(a.pred, name(self.world, &a.subj));
        let text = templates::question(b.pred, &via, self.held_out);
        Some(Draft {
            shape: Primitive::Chain,
            text,
            answer: render(self.world, &b.obj),
            facts: vec![a, b],
        })
    }

    fn compare<R: Rng>(&self, answerable: bool, rng: &mut R) -> Option<Draft<'a>> {
        // a.id < b.id: each unordered pair once; the display order is drawn below.
        let (mut a, mut b) = self.pair(answerable, rng, |a, b| {
            a.pred == b.pred && a.pred.target().is_none() && a.id < b.id && self.same_kind(a, b)
        })?;
        if rng.random_bool(0.5) {
            std::mem::swap(&mut a, &mut b);
        }
        let (na, nb) = (name(self.world, &a.subj), name(self.world, &b.subj));
        let winner = if first_wins(a, b) { na } else { nb };
        let text = templates::comparison(a.pred, na, nb, self.held_out);
        Some(Draft {
            shape: Primitive::Compare,
            text,
            answer: winner.to_string(),
            facts: vec![a, b],
        })
    }

    /// A random pair of world facts matching `fits`, both stated if
    /// `answerable`, else with at least one unstated.
    fn pair<R: Rng>(
        &self,
        answerable: bool,
        rng: &mut R,
        fits: impl Fn(&Fact, &Fact) -> bool,
    ) -> Option<(&'a Fact, &'a Fact)> {
        let facts = &self.world.facts;
        let pairs: Vec<(&Fact, &Fact)> = facts
            .iter()
            .flat_map(|a| facts.iter().map(move |b| (a, b)))
            .filter(|(a, b)| fits(a, b) && (self.stated(a) && self.stated(b)) == answerable)
            .collect();
        pairs.choose(rng).copied()
    }

    fn same_kind(&self, a: &Fact, b: &Fact) -> bool {
        let kind = |f: &Fact| self.world.entity(&f.subj).map(|e| e.kind);
        kind(a) == kind(b)
    }

    fn finish(&self, d: Draft, answerable: bool) -> Question {
        let hops = if d.shape == Primitive::Chain { 2 } else { 1 };
        if answerable {
            let ids = d.facts.iter().map(|f| f.id.clone()).collect();
            let sources = d
                .facts
                .iter()
                .map(|f| self.source_of[f.id.as_str()].to_string())
                .collect();
            return Question::answerable_with(d.text, d.shape, hops, d.answer, ids, sources);
        }
        let support = d.facts.iter().map(|f| f.id.clone()).collect();
        let unstated: Vec<&Fact> = d.facts.into_iter().filter(|f| !self.stated(f)).collect();
        let what: Vec<String> = unstated
            .iter()
            .map(|f| templates::describe(f.pred, name(self.world, &f.subj)))
            .collect();
        let missing = format!("no source mentions {}", what.join(" or "));
        Question::unanswerable(
            d.text,
            d.shape,
            hops,
            missing,
            unstated[0].id.clone(),
            support,
        )
    }
}

/// Whether `a` is the answer of the comparison (see `Pred::larger_wins`).
fn first_wins(a: &Fact, b: &Fact) -> bool {
    let (Obj::Int(x), Obj::Int(y)) = (&a.obj, &b.obj) else {
        unreachable!("compare only draws numeric facts");
    };
    if a.pred.larger_wins() { x > y } else { x < y }
}
