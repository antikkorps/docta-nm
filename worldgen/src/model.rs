//! Ground-truth types of a world: entities and facts as triples.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    City,
    Guild,
    Person,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::City => "city",
            Kind::Guild => "guild",
            Kind::Person => "person",
        }
    }

    /// The predicates every entity of this kind has, in generation order.
    /// Shared by the generator's tests and the question layer.
    pub fn preds(self) -> &'static [Pred] {
        match self {
            Kind::City => &[Pred::FoundedIn, Pred::Population],
            Kind::Guild => &[Pred::FoundedIn, Pred::BasedIn],
            Kind::Person => &[Pred::BornIn, Pred::LivesIn, Pred::MemberOf],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pred {
    FoundedIn,
    Population,
    BasedIn,
    BornIn,
    LivesIn,
    MemberOf,
}

impl Pred {
    pub fn as_str(self) -> &'static str {
        match self {
            Pred::FoundedIn => "founded_in",
            Pred::Population => "population",
            Pred::BasedIn => "based_in",
            Pred::BornIn => "born_in",
            Pred::LivesIn => "lives_in",
            Pred::MemberOf => "member_of",
        }
    }

    /// Kind of the entity the object points to; `None` for a literal number.
    pub fn target(self) -> Option<Kind> {
        match self {
            Pred::BasedIn | Pred::LivesIn => Some(Kind::City),
            Pred::MemberOf => Some(Kind::Guild),
            Pred::FoundedIn | Pred::Population | Pred::BornIn => None,
        }
    }
}

/// Object of a fact: another entity of the same world, or a literal number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Obj {
    Entity(String),
    Int(i64),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entity {
    pub id: String,
    pub kind: Kind,
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub id: String,
    pub subj: String,
    pub pred: Pred,
    pub obj: Obj,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct World {
    pub world_id: String,
    pub seed: u64,
    pub entities: Vec<Entity>,
    pub facts: Vec<Fact>,
}

impl World {
    pub fn entity(&self, id: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// The object of `subj`'s fact with predicate `pred`, if any.
    pub fn object(&self, subj: &str, pred: Pred) -> Option<&Obj> {
        self.facts
            .iter()
            .find(|f| f.subj == subj && f.pred == pred)
            .map(|f| &f.obj)
    }
}
