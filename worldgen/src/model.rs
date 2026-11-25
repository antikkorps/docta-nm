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
            Kind::City => &[Pred::FoundedIn, Pred::Population, Pred::Elevation],
            Kind::Guild => &[Pred::FoundedIn, Pred::BasedIn, Pred::Members],
            Kind::Person => &[Pred::BornIn, Pred::LivesIn, Pred::MemberOf, Pred::Height],
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
    Elevation,
    Members,
    Height,
}

impl Pred {
    pub const ALL: [Pred; 9] = [
        Pred::FoundedIn,
        Pred::Population,
        Pred::BasedIn,
        Pred::BornIn,
        Pred::LivesIn,
        Pred::MemberOf,
        Pred::Elevation,
        Pred::Members,
        Pred::Height,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Pred::FoundedIn => "founded_in",
            Pred::Population => "population",
            Pred::BasedIn => "based_in",
            Pred::BornIn => "born_in",
            Pred::LivesIn => "lives_in",
            Pred::MemberOf => "member_of",
            Pred::Elevation => "elevation",
            Pred::Members => "members",
            Pred::Height => "height",
        }
    }

    /// Kind of the entity the object points to; `None` for a literal number.
    pub fn target(self) -> Option<Kind> {
        match self {
            Pred::BasedIn | Pred::LivesIn => Some(Kind::City),
            Pred::MemberOf => Some(Kind::Guild),
            _ => None,
        }
    }

    /// In a comparison, whether the larger value answers it: years ask for the
    /// earlier one, every other quantity for the larger.
    pub fn larger_wins(self) -> bool {
        !matches!(self, Pred::FoundedIn | Pred::BornIn)
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

    pub fn fact(&self, id: &str) -> Option<&Fact> {
        self.facts.iter().find(|f| f.id == id)
    }

    /// The object of `subj`'s fact with predicate `pred`, if any.
    pub fn object(&self, subj: &str, pred: Pred) -> Option<&Obj> {
        self.facts
            .iter()
            .find(|f| f.subj == subj && f.pred == pred)
            .map(|f| &f.obj)
    }
}
