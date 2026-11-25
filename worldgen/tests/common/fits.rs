//! Whether a text could have been written from a template: the matchers the
//! ood guards use.

/// Whether `text` could have been written from `template`: its literal pieces
/// in order, each placeholder filled by a name or a number, as worldgen fills
/// them. A looser match would let "{o}" swallow a frame ("It is known that X")
/// and see a held-out form inside a seen one.
pub fn fits(template: &str, text: &str) -> bool {
    let parts: Vec<&str> = template.split(['{', '}']).collect();
    // parts alternate: literal, placeholder, literal, ...
    let literals: Vec<&str> = parts.iter().step_by(2).copied().collect();
    text.strip_prefix(literals[0])
        .is_some_and(|rest| fill_slots(&literals[1..], rest))
}

/// Matches `text` as filler, literal, filler, literal, ... to the end.
/// Literal pieces in order, placeholders filled by anything: a question's
/// subject may be a whole chain link ("the guild X is a member of").
pub fn fits_loose(template: &str, text: &str) -> bool {
    let pieces: Vec<&str> = template.split(['{', '}']).step_by(2).collect();
    let (first, last) = (pieces[0], pieces[pieces.len() - 1]);
    if !text.starts_with(first) || !text.ends_with(last) || text.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for piece in &pieces[1..pieces.len() - 1] {
        match rest.find(piece) {
            Some(at) => rest = &rest[at + piece.len()..],
            None => return false,
        }
    }
    true
}

pub fn fill_slots(literals: &[&str], text: &str) -> bool {
    let Some((literal, rest_literals)) = literals.split_first() else {
        return text.is_empty();
    };
    text.match_indices(literal).any(|(at, _)| {
        is_filler(&text[..at]) && fill_slots(rest_literals, &text[at + literal.len()..])
    }) || (literal.is_empty() && rest_literals.is_empty() && is_filler(text))
}

/// A name ("Saesbron Kindro") or a number.
pub fn is_filler(text: &str) -> bool {
    !text.is_empty()
        && (text.chars().all(|c| c.is_ascii_digit())
            || text.split(' ').all(|w| {
                let mut c = w.chars();
                c.next().is_some_and(|f| f.is_ascii_uppercase())
                    && c.all(|r| r.is_ascii_lowercase())
            }))
}

/// Whether `template`'s literal pieces appear in order anywhere in `text`:
/// a chain link sits inside a question.
pub fn holds(template: &str, text: &str) -> bool {
    let mut rest = text;
    for piece in template
        .split(['{', '}'])
        .step_by(2)
        .filter(|p| !p.is_empty())
    {
        match rest.find(piece) {
            Some(at) => rest = &rest[at + piece.len()..],
            None => return false,
        }
    }
    true
}

/// A sentence's structure: placeholders, punctuation and a few function words
/// kept, prepositions merged, any run of other words one slot.
pub fn skeleton(template: &str) -> Vec<String> {
    const KEPT: [&str; 10] = [
        "is", "the", "of", "was", "a", "an", "has", "its", "as", "are",
    ];
    let mut out: Vec<String> = Vec::new();
    let mut rest = template;
    while let Some(c) = rest.chars().next() {
        let token = if c == '{' {
            let end = rest.find('}').expect("placeholders are closed");
            rest[..=end].to_string()
        } else if c.is_ascii_alphabetic() {
            rest.split(|c: char| !c.is_ascii_alphabetic())
                .next()
                .unwrap_or("")
                .to_string()
        } else {
            c.to_string()
        };
        rest = &rest[token.len()..];
        let lower = token.to_lowercase();
        let slot = match lower.as_str() {
            " " => continue,
            "in" | "to" | "at" | "on" => "P".to_string(),
            w if w.chars().all(|c| c.is_ascii_alphabetic()) && !KEPT.contains(&w) => "w".into(),
            _ => lower,
        };
        if !(slot == "w" && out.last().is_some_and(|l| l == "w")) {
            out.push(slot);
        }
    }
    out
}
