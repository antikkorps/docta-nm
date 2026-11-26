//! Pronouns: a later sentence of a source may name its subject by pronoun
//! ("She lives in Brask"), as real text does at every turn ("he", "his" and
//! "her" are among the commonest words the cores did not know). A pronoun
//! always follows a sentence of the same source that names its subject.

use crate::frames::PARENTHETICALS;
use crate::model::{Entity, Kind, Pred};

/// How a person is referred to, drawn once per person from the world seed and
/// its id: never from the name, so a core must follow the text, not guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pronoun {
    He,
    She,
    /// Singular "they", as for a person who uses it: the verb is plural.
    They,
    It,
}

impl Pronoun {
    /// 45% he, 45% she, 10% they for a person; "it" for a city or a guild.
    /// A hash, not the corpus RNG, so the draw shifts no other draw.
    pub fn of(seed: u64, entity: &Entity) -> Pronoun {
        if entity.kind != Kind::Person {
            return Pronoun::It;
        }
        match fnv(&format!("{seed}:{}", entity.id)) % 20 {
            0..=8 => Pronoun::He,
            9..=17 => Pronoun::She,
            _ => Pronoun::They,
        }
    }

    pub const ALL: [Pronoun; 4] = [Pronoun::He, Pronoun::She, Pronoun::They, Pronoun::It];

    fn subject(self) -> &'static str {
        ["he", "she", "they", "it"][self as usize]
    }

    fn object(self) -> &'static str {
        ["him", "her", "them", "it"][self as usize]
    }

    fn possessive(self) -> &'static str {
        ["his", "her", "their", "its"][self as usize]
    }
}

/// Every word a pronoun form writes: never a name (crate::names).
pub const WORDS: [&str; 9] = [
    "he", "she", "they", "it", "him", "her", "them", "his", "their",
];

/// FNV-1a: stable across platforms and Rust versions, unlike the std hasher.
fn fnv(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// Words a frame puts right before the clause's subject: there the
/// placeholder is the subject of the sentence.
const BEFORE_SUBJECT: [&str; 5] = ["that ", ", ", ": ", "say ", "year "];

/// `template` with its subject named by `p`, or `None` where no pronoun
/// reads naturally (after the verb: "At 210 metres lies {s}"; after "in").
pub fn variant(template: &str, p: Pronoun) -> Option<String> {
    let at = template.find("{s}")?;
    let (before, after) = (&template[..at], &template[at + 3..]);
    // "It, it is recorded, ..." and "Few would dispute it: it ..." stutter.
    if p == Pronoun::It && (says_it(before) || says_it(after)) {
        return None;
    }
    // An aside may follow a pronoun ("She, as is well known, ..."), a
    // description may not ("She, a name found in many records, ...").
    let opens = after.starts_with(' ') || PARENTHETICALS.iter().any(|a| after.starts_with(a));
    if before.is_empty() && opens {
        return Some(capital(&subject(p, after)));
    }
    if BEFORE_SUBJECT.iter().any(|w| before.ends_with(w)) && opens {
        return Some(format!("{before}{}", subject(p, after)));
    }
    if before.ends_with("to ") || before.ends_with("has ") {
        return Some(format!("{before}{}{after}", p.object()));
    }
    possessive(before, after, p)
}

fn says_it(text: &str) -> bool {
    text.split(|c: char| !c.is_ascii_alphabetic())
        .any(|w| w.eq_ignore_ascii_case("it"))
}

/// "{s} lives in" -> "she lives in", "they live in": a singular "they" takes
/// a plural verb, found after an aside if one follows the subject.
fn subject(p: Pronoun, after: &str) -> String {
    if p != Pronoun::They {
        return format!("{}{after}", p.subject());
    }
    // ", as is well known, lives in" -> the aside, then " lives in".
    let aside = match after.strip_prefix(", ") {
        Some(rest) => rest.find(", ").map_or(0, |end| end + 3),
        None => 0,
    };
    let (head, rest) = after.split_at(aside);
    let rest = rest.trim_start();
    let (verb, tail) = rest.split_once(' ').unwrap_or((rest, ""));
    format!("they{head} {} {tail}", plural(verb))
        .trim_end()
        .to_string()
}

fn plural(verb: &str) -> String {
    match verb {
        "is" => "are".into(),
        "was" => "were".into(),
        "has" => "have".into(),
        "does" => "do".into(),
        v if ["ches", "shes", "sses", "xes"]
            .iter()
            .any(|e| v.ends_with(e)) =>
        {
            v[..v.len() - 2].into()
        }
        v if v.ends_with('s') => v[..v.len() - 1].into(),
        v => v.into(),
    }
}

/// "The home of {s} is" -> "Her home is"; "fill the ranks of {s}" -> "fill
/// its ranks".
fn possessive(before: &str, after: &str, p: Pronoun) -> Option<String> {
    let head = before.strip_suffix(" of ")?;
    let the = head.rfind("the ").or_else(|| head.rfind("The "))?;
    let noun = &head[the + 4..];
    if noun.contains(" of ") {
        return None; // "Its number of members" reads worse than the name
    }
    let word = if head[the..].starts_with('T') {
        capital(p.possessive())
    } else {
        p.possessive().to_string()
    };
    Some(format!("{}{word} {noun}{after}", &head[..the]))
}

fn capital(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map(|c| c.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

/// Every pronoun form a source can write for `template` of `pred`: what the
/// rules and the seal tests must recognise as the same phrasing. A person's
/// predicate takes he, she or they; a city's or a guild's takes it.
pub fn variants(template: &str, pred: Pred) -> Vec<String> {
    let people = Kind::Person.preds().contains(&pred);
    Pronoun::ALL
        .into_iter()
        .filter(|&p| people == (p != Pronoun::It))
        .filter_map(|p| variant(template, p))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_subject_becomes_a_pronoun_with_its_verb_agreeing() {
        let t = "{s} lives in {o}.";
        assert_eq!(variant(t, Pronoun::She).unwrap(), "She lives in {o}.");
        assert_eq!(variant(t, Pronoun::They).unwrap(), "They live in {o}.");
        let t = "{s} was born in {o}.";
        assert_eq!(variant(t, Pronoun::They).unwrap(), "They were born in {o}.");
        let t = "{s} reaches a height of {o} centimetres.";
        assert_eq!(
            variant(t, Pronoun::They).unwrap(),
            "They reach a height of {o} centimetres."
        );
    }

    #[test]
    fn after_an_aside_the_verb_still_agrees() {
        let t = "{s}, as is well known, lives in {o}.";
        assert_eq!(
            variant(t, Pronoun::He).unwrap(),
            "He, as is well known, lives in {o}."
        );
        assert_eq!(
            variant(t, Pronoun::They).unwrap(),
            "They, as is well known, live in {o}."
        );
        let t = "It is said that {s}, by all accounts, was born in {o}.";
        assert_eq!(
            variant(t, Pronoun::They).unwrap(),
            "It is said that they, by all accounts, were born in {o}."
        );
    }

    #[test]
    fn inside_a_frame_the_pronoun_stays_lower_case() {
        let t = "Records show that {s} has existed since {o}.";
        assert_eq!(
            variant(t, Pronoun::It).unwrap(),
            "Records show that it has existed since {o}."
        );
        let t = "{o} is the year {s} was founded.";
        assert_eq!(
            variant(t, Pronoun::It).unwrap(),
            "{o} is the year it was founded."
        );
    }

    #[test]
    fn an_object_and_an_owner_take_their_own_forms() {
        assert_eq!(
            variant("{o} is home to {s}.", Pronoun::He).unwrap(),
            "{o} is home to him."
        );
        assert_eq!(
            variant("{o} has {s} as a member.", Pronoun::They).unwrap(),
            "{o} has them as a member."
        );
        assert_eq!(
            variant("The home of {s} is {o}.", Pronoun::She).unwrap(),
            "Her home is {o}."
        );
        assert_eq!(
            variant("{o} fellows fill the ranks of {s}.", Pronoun::It).unwrap(),
            "{o} fellows fill its ranks."
        );
    }

    #[test]
    fn no_pronoun_where_it_would_read_wrong() {
        assert_eq!(
            variant("At {o} metres above sea level lies {s}.", Pronoun::It),
            None
        );
        assert_eq!(variant("{o} inhabitants live in {s}.", Pronoun::It), None);
        assert_eq!(
            variant("{s}, it is recorded, was founded in {o}.", Pronoun::It),
            None
        );
        assert_eq!(
            variant("Few would dispute it: {s} was founded in {o}.", Pronoun::It),
            None
        );
        assert_eq!(
            variant("It is said that {s} was founded in {o}.", Pronoun::It),
            None
        );
    }

    #[test]
    fn a_description_never_follows_a_pronoun() {
        // "She, a name found in many records, ..." would call a person a name.
        let t = "{s}, a name found in many records, was born in {o}.";
        assert_eq!(variant(t, Pronoun::She), None);
        assert_eq!(
            variant(
                "{s}, a town of some renown, counts {o} people.",
                Pronoun::It
            ),
            None
        );
    }

    #[test]
    fn a_long_noun_keeps_the_name() {
        assert_eq!(
            variant("The number of members of {s} is {o}.", Pronoun::It),
            None
        );
    }

    #[test]
    fn only_the_pronouns_a_predicate_s_subject_takes() {
        use crate::model::Pred;
        let people = variants("{s} was born in {o}.", Pred::BornIn);
        assert!(people.iter().all(|v| !v.starts_with("It ")) && people.len() == 3);
        assert_eq!(
            variants("{s} was founded in {o}.", Pred::FoundedIn),
            vec!["It was founded in {o}."]
        );
    }

    #[test]
    fn people_get_he_she_and_they_about_45_45_10_and_places_it() {
        let person = |i: usize| Entity {
            id: format!("e{i}"),
            kind: Kind::Person,
            name: "N".into(),
        };
        let drawn: Vec<Pronoun> = (0..2000).map(|i| Pronoun::of(42, &person(i))).collect();
        let share = |p| drawn.iter().filter(|&&d| d == p).count() as f64 / 2000.0;
        assert!((share(Pronoun::He) - 0.45).abs() < 0.04);
        assert!((share(Pronoun::She) - 0.45).abs() < 0.04);
        assert!((share(Pronoun::They) - 0.10).abs() < 0.03);
        let city = Entity {
            id: "e1".into(),
            kind: Kind::City,
            name: "N".into(),
        };
        assert_eq!(Pronoun::of(42, &city), Pronoun::It);
    }
}
