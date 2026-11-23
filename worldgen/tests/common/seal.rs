//! What the ood and val seals share: the predicates, the sentences a corpus
//! writes, and template matching over a list.

use super::fits::{fits, fits_loose};
use docta_worldgen::corpus::Corpus;
use docta_worldgen::model::Pred;
use std::collections::BTreeSet;

pub const PREDS: [Pred; 9] = [
    Pred::FoundedIn,
    Pred::Population,
    Pred::BasedIn,
    Pred::BornIn,
    Pred::LivesIn,
    Pred::MemberOf,
    Pred::Elevation,
    Pred::Members,
    Pred::Height,
]; // fmt: skip

/// Every sentence a source writes itself. A copy repeats its origin's
/// sentence behind an attribution; tests/traps.rs seals the attributions.
pub fn sentences(c: &Corpus) -> Vec<String> {
    c.sources
        .iter()
        .filter(|s| s.copy_of.is_none())
        .flat_map(|s| {
            s.text
                .split(". ")
                .map(|x| x.trim_end_matches('.').to_string() + ".")
        })
        .collect()
}

pub fn any_fits(templates: &[&str], text: &str) -> bool {
    templates.iter().any(|t| fits(t, text))
}

pub fn any_fits_loose(templates: &[&str], text: &str) -> bool {
    templates.iter().any(|t| fits_loose(t, text))
}

/// Lowercased words of templates, placeholders left out.
pub fn words<'a>(texts: impl IntoIterator<Item = &'a str>) -> BTreeSet<String> {
    texts
        .into_iter()
        .flat_map(|t| t.split(|c: char| !c.is_ascii_alphabetic()))
        .filter(|w| !w.is_empty() && *w != "s" && *w != "o" && *w != "t" && *w != "x")
        .map(str::to_lowercase)
        .collect()
}
