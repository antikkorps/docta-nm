//! Writes a full dataset: one JSONL file per split (train, val, test, ood).

use crate::corpus::{Options, build};
use crate::jsonl::lines;
use crate::seed::world_seed;
use crate::split::{Ratios, Split, assign};
use crate::world::generate;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub seed: u64,
    pub worlds: u64,
    pub ratios: Ratios,
    pub options: Options,
}

/// What a split holds, as the binary reports it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    pub worlds: u64,
    pub questions: u64,
    /// `--trap-ratio` is a ceiling (one trap per true source, none on a fact
    /// another question uses): this is how many questions really got one.
    pub traps: u64,
}

/// Writes every world to the file of its split; returns what each split holds.
pub fn write(cfg: &Config, out: &Path) -> io::Result<BTreeMap<Split, Counts>> {
    fs::create_dir_all(out)?;
    let mut files = BTreeMap::new();
    for split in Split::ALL {
        let file = File::create(out.join(format!("{}.jsonl", split.as_str())))?;
        files.insert(split, BufWriter::new(file));
    }
    let mut counts: BTreeMap<Split, Counts> =
        Split::ALL.iter().map(|s| (*s, Counts::default())).collect();
    for i in 0..cfg.worlds {
        let seed = world_seed(cfg.seed, i);
        let split = assign(seed, &cfg.ratios);
        let corpus = build(generate(&format!("w_{i:06}"), seed), split, &cfg.options);
        let file = files.get_mut(&split).expect("one file per split");
        for line in lines(&corpus, split) {
            writeln!(file, "{line}")?;
        }
        let count = counts.get_mut(&split).expect("one count per split");
        count.worlds += 1;
        count.questions += corpus.questions.len() as u64;
        count.traps += corpus.questions.iter().filter(|q| q.trap.is_some()).count() as u64;
    }
    for file in files.values_mut() {
        file.flush()?;
    }
    Ok(counts)
}
