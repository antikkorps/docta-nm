//! What a source wraps around a clause: the frame that says where the fact
//! comes from, and the aside set after its subject. Words of their own
//! ("archives", "chronicles", "widely"), never a name's.
//!
//! None opens with "According to", says "reports", or puts a source noun
//! before "<verb> that" as "{t} records that" does: those are how a copy
//! cites its origin (crate::liars), which Confront reads.

/// Structural frames any clause takes; `{c}` is the clause without its stop.
pub const FRAMES: [&str; 30] = [
    "{c}.",
    "Records show that {c}.",
    "{c}, the records say.",
    "It is said that {c}.",
    "As far as anyone knows, {c}.",
    "The records say that {c}.",
    "{c}, as far as anyone knows.",
    "It is known that {c}.",
    "In the archives, {c}.",
    "{c}, the chronicles say.",
    "{c}, as the registers state.",
    "All accounts agree: {c}.",
    "{c}, old letters tell.",
    "{c}, as the archives note.",
    "By all accounts, {c}.",
    "It is widely held that {c}.",
    "{c}, scholars agree.",
    "By local tradition, {c}.",
    "{c}, every guide to the region notes.",
    "To this day, people say {c}.",
    "{c}, so the elders say.",
    "{c}, travellers write.",
    "As every schoolchild learns, {c}.",
    "Whatever else is doubted, {c}.",
    "Few would dispute it: {c}.",
    "Beyond any doubt, {c}.",
    "{c}, or so the story goes.",
    "{c}, as the old songs have it.",
    "{c}, the old maps show.",
    "Ask anyone in the region: {c}.",
];

/// Asides set after a clause's opening subject, so a word never directly
/// follows it.
pub const PARENTHETICALS: [&str; 8] = [
    ", it is recorded,",
    ", it is said,",
    ", as the chronicles note,",
    ", by all accounts,",
    ", as scholars agree,",
    ", so the elders tell,",
    ", if the sources are right,",
    ", as is well known,",
];
