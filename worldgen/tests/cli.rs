//! End-to-end checks on the files the binary writes.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const SPLITS: [&str; 4] = ["train", "val", "test", "ood"];

fn run(dir: &Path, extra: &[&str]) -> bool {
    Command::new(env!("CARGO_BIN_EXE_docta-worldgen"))
        .args(["--seed", "42", "--worlds", "300", "--out"])
        .arg(dir)
        .args(extra)
        .status()
        .expect("binary runs")
        .success()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("docta-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn read(dir: &Path, split: &str) -> String {
    std::fs::read_to_string(dir.join(format!("{split}.jsonl"))).expect("split file exists")
}

#[test]
fn same_seed_gives_byte_identical_files() {
    let (a, b) = (temp_dir("a"), temp_dir("b"));
    assert!(run(&a, &[]) && run(&b, &[]));
    for split in SPLITS {
        assert_eq!(read(&a, split), read(&b, split), "{split}.jsonl differs");
    }
}

#[test]
fn every_line_is_v1_json_and_worlds_never_cross_splits() {
    let dir = temp_dir("lines");
    assert!(run(&dir, &[]));
    let mut owner: BTreeMap<String, &str> = BTreeMap::new();
    for split in SPLITS {
        let mut declared = BTreeSet::new();
        for line in read(&dir, split).lines() {
            let v: Value = serde_json::from_str(line).expect("valid JSON");
            assert_eq!(v["schema"], "v1");
            let id = v["world_id"].as_str().expect("world_id").to_string();
            match v["type"].as_str() {
                Some("world") => {
                    assert_eq!(v["split"], split);
                    assert!(
                        owner.insert(id.clone(), split).is_none(),
                        "{id} in two splits"
                    );
                    declared.insert(id);
                }
                Some("source" | "question") => {
                    assert!(declared.contains(&id), "{id} before its world")
                }
                other => panic!("unknown record type {other:?}"),
            }
        }
    }
    assert_eq!(owner.len(), 300);
}

#[test]
fn invalid_options_fail() {
    let dir = temp_dir("bad");
    assert!(!run(&dir, &["--split-ratio", "0.5,0.1"]));
    assert!(!run(&dir, &["--unanswerable-ratio", "1.5"]));
    assert!(!run(&dir, &["--max-hops", "3"]));
}
