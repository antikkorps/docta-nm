//! Generates a whole dataset into `train.jsonl`, `val.jsonl`, `test.jsonl` and `ood.jsonl`.

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

/// Writes every world to the file of its split; returns worlds per split.
pub fn write(cfg: &Config, out: &Path) -> io::Result<BTreeMap<Split, u64>> {
    fs::create_dir_all(out)?;
    let mut files = BTreeMap::new();
    for split in Split::ALL {
        let file = File::create(out.join(format!("{}.jsonl", split.as_str())))?;
        files.insert(split, BufWriter::new(file));
    }
    let mut counts: BTreeMap<Split, u64> = Split::ALL.iter().map(|s| (*s, 0)).collect();
    for i in 0..cfg.worlds {
        let seed = world_seed(cfg.seed, i);
        let split = assign(seed, &cfg.ratios);
        let corpus = build(generate(&format!("w_{i:06}"), seed), split, &cfg.options);
        let file = files.get_mut(&split).expect("one file per split");
        for line in lines(&corpus, split) {
            writeln!(file, "{line}")?;
        }
        *counts.get_mut(&split).expect("one count per split") += 1;
    }
    for file in files.values_mut() {
        file.flush()?;
    }
    Ok(counts)
}
