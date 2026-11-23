//! Traps (JSONL v1): disagreeing sources around the fact a question asks about.
//!
//! The challenge step of the method needs something to challenge: a false
//! source to outweigh, copies of one false origin to count as one, or two
//! equally reliable sources the method must not settle. Traps add sources and
//! never remove the true one; reliability helps tell them apart without being
//! enough on its own.

use crate::corpus::{Primitive, Question, Source, Trap};
use crate::liars::{Claim, Titles, cents};
use crate::model::{Fact, World};
use crate::sources::name;
use crate::split::Wording;
use crate::{questions as ask, templates};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, RngExt};
use std::collections::{BTreeMap, BTreeSet};

const KINDS: [Trap; 3] = [Trap::Contradiction, Trap::FalseConsensus, Trap::Undecidable];

/// What a trap reads from the world it is set in.
pub struct Ground<'a> {
    pub world: &'a World,
    /// The facts no source states (one per entity).
    pub withheld: &'a BTreeSet<String>,
    pub wording: Wording,
}

/// Sets a trap on `round(ratio × answerable retrieve questions)` questions.
pub fn apply<R: Rng>(
    ground: &Ground,
    sources: &mut Vec<Source>,
    questions: &mut [Question],
    ratio: f64,
    rng: &mut R,
) {
    let picked = pick(questions, ratio, rng);
    if picked.is_empty() {
        return;
    }
    let (mut titles, mut contested) = (Titles::of(sources), BTreeSet::new());
    let mut added = Vec::new();
    for i in picked {
        contested.insert(questions[i].required_facts[0].clone());
        let kind = choose_kind(ground, sources, questions, &contested, rng);
        let q = &mut questions[i];
        let how = (ground.wording, &mut titles);
        added.extend(set(ground.world, sources, q, kind, how, rng));
    }
    insert_and_renumber(sources, added, questions, rng);
}

/// An undecidable question is unanswerable, so it takes an unanswerable
/// retrieve slot of its world, which becomes a fresh answerable retrieve
/// question: the share of unanswerable questions, overall and per shape, stays
/// as designed. Without such a slot, the trap is one of the decidable kinds.
fn choose_kind<R: Rng>(
    ground: &Ground,
    sources: &[Source],
    questions: &mut [Question],
    contested: &BTreeSet<String>,
    rng: &mut R,
) -> Trap {
    let kind = *KINDS.choose(rng).expect("three kinds");
    if kind != Trap::Undecidable || swap_slot(ground, sources, questions, contested, rng) {
        return kind;
    }
    *[Trap::Contradiction, Trap::FalseConsensus]
        .choose(rng)
        .expect("two kinds")
}

fn swap_slot<R: Rng>(
    ground: &Ground,
    sources: &[Source],
    questions: &mut [Question],
    contested: &BTreeSet<String>,
    rng: &mut R,
) -> bool {
    let slot = questions
        .iter()
        .position(|q| q.absent_fact.is_some() && q.shape == Primitive::Retrieve);
    let Some(k) = slot else { return false };
    let (world, hidden) = (ground.world, ground.withheld);
    let asked = (&*questions, contested);
    let Some(mut fresh) = ask::fresh_retrieve(world, sources, hidden, ground.wording, asked, rng)
    else {
        return false;
    };
    fresh.id = questions[k].id.clone();
    questions[k] = fresh;
    true
}

fn pick<R: Rng>(questions: &[Question], ratio: f64, rng: &mut R) -> Vec<usize> {
    let mut eligible: Vec<usize> = (0..questions.len())
        .filter(|&i| questions[i].primitive == Primitive::Retrieve && questions[i].answerable())
        .filter(|&i| sole_user(questions, i))
        .collect();
    let count = (eligible.len() as f64 * ratio).round() as usize;
    eligible.shuffle(rng);
    // One trap per true source: a second trap would reset the reliability the
    // first one depends on (an undecidable pair must stay within 0.05).
    let mut used = BTreeSet::new();
    let mut picked: Vec<usize> = eligible
        .into_iter()
        .filter(|&i| used.insert(questions[i].required_sources[0].clone()))
        .take(count)
        .collect();
    picked.sort_unstable();
    picked
}

/// Whether no other question rests on question `i`'s fact: a comparison or a
/// chain through a contested fact would inherit a conflict it does not declare.
fn sole_user(questions: &[Question], i: usize) -> bool {
    let fact = &questions[i].required_facts[0];
    questions
        .iter()
        .enumerate()
        .all(|(j, q)| j == i || !q.support.contains(fact))
}

/// The trap's new sources, with temporary ids; the question learns its trap.
fn set<R: Rng>(
    world: &World,
    sources: &mut [Source],
    q: &mut Question,
    kind: Trap,
    (wording, titles): (Wording, &mut Titles),
    rng: &mut R,
) -> Vec<Source> {
    let fact = world.fact(&q.required_facts[0]).expect("fact");
    let truth = sources.iter_mut().find(|s| s.id == q.required_sources[0]);
    let truth = truth.expect("source");
    let tag = format!("{}:{}", q.text, kind.as_str());
    let mut claim = Claim::new(world, fact, tag, wording, titles, rng);
    q.trap = Some(kind);
    match kind {
        Trap::Contradiction => contradiction(&mut claim, truth, rng),
        Trap::FalseConsensus => false_consensus(&mut claim, truth, rng),
        Trap::Undecidable => {
            let other = equal_rival(&mut claim, truth, rng);
            undecidable(q, world, fact);
            vec![other]
        }
    }
}

/// The true source becomes clearly more reliable than its single rival.
fn contradiction<R: Rng>(claim: &mut Claim, truth: &mut Source, rng: &mut R) -> Vec<Source> {
    // Every source, false or not, stays within 0.60–1.00: a value below the
    // truthful range would give a liar away without reading.
    truth.reliability = cents(rng, 80, 100);
    let ceiling = (truth.reliability * 100.0).round() as u8 - 20;
    vec![claim.origin(cents(rng, 60, ceiling), rng)]
}

/// A weak false origin and its copies outnumber one reliable primary source.
fn false_consensus<R: Rng>(claim: &mut Claim, truth: &mut Source, rng: &mut R) -> Vec<Source> {
    truth.reliability = cents(rng, 85, 100);
    // At least 0.10 below the truth (0.85+): beyond the 0.05 of an undecidable pair.
    let origin = claim.origin(cents(rng, 60, 75), rng);
    let copies: Vec<Source> = (0..rng.random_range(2..=4))
        .map(|k| claim.copy(&origin, k, rng))
        .collect();
    [vec![origin], copies].concat()
}

/// A rival within 0.05 of the true source: nothing allows choosing.
fn equal_rival<R: Rng>(claim: &mut Claim, truth: &Source, rng: &mut R) -> Source {
    let t = (truth.reliability * 100.0).round() as u8;
    claim.origin(
        cents(rng, t.saturating_sub(5).max(60), (t + 5).min(100)),
        rng,
    )
}

fn undecidable(q: &mut Question, world: &World, fact: &Fact) {
    let about = templates::describe(fact.pred, name(world, &fact.subj));
    // Unanswerable like any other: spot_missing, nothing required, its shape
    // (retrieve) only visible through `shape`.
    q.primitive = Primitive::SpotMissing;
    q.answer = None;
    q.missing = Some(format!("two equally reliable sources disagree on {about}"));
    q.required_facts.clear();
    q.required_sources.clear();
    q.expected_exit = Some("unresolved_contradiction");
}

/// New sources go to random places, then every source is renumbered s1…sn,
/// so neither an id nor a position tells a trap source apart.
fn insert_and_renumber<R: Rng>(
    sources: &mut Vec<Source>,
    added: Vec<Source>,
    questions: &mut [Question],
    rng: &mut R,
) {
    for s in added {
        let at = rng.random_range(0..=sources.len());
        sources.insert(at, s);
    }
    let ids: BTreeMap<String, String> = sources
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.clone(), format!("s{}", i + 1)))
        .collect();
    for s in sources.iter_mut() {
        s.id = ids[&s.id].clone();
        s.copy_of = s.copy_of.as_ref().map(|o| ids[o].clone());
    }
    for q in questions.iter_mut() {
        q.required_sources = q.required_sources.iter().map(|s| ids[s].clone()).collect();
    }
}
