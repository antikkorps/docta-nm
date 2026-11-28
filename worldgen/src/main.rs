use clap::Parser;
use docta_worldgen::corpus::Options;
use docta_worldgen::dataset::{Config, write};
use docta_worldgen::split::Ratios;
use std::path::PathBuf;
use std::process::ExitCode;

/// Generates fictional worlds, their sources and questions as JSONL v2.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Global seed; every world seed derives from it.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of worlds to generate.
    #[arg(long, default_value_t = 1000)]
    worlds: u64,
    /// Output directory for train.jsonl, val.jsonl, test.jsonl and ood.jsonl.
    #[arg(long, default_value = "data")]
    out: PathBuf,
    /// Share of worlds per split, as train,val,test,ood.
    #[arg(long, default_value = "0.7,0.1,0.1,0.1", value_parser = Ratios::parse)]
    split_ratio: Ratios,
    /// Share of questions per world that no source can answer.
    #[arg(long, default_value_t = 0.33, value_parser = parse_ratio)]
    unanswerable_ratio: f64,
    /// Maximum reasoning hops per question.
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u8).range(1..=2))]
    max_hops: u8,
    /// Ceiling on the share of answerable retrieve questions whose fact gets
    /// a trap (contradiction, false consensus or undecidable): one trap per
    /// true source and none on a fact another question uses, so fewer get one
    /// (about 23% at 0.3). The real count is reported per split.
    #[arg(long, default_value_t = 0.3, value_parser = parse_ratio)]
    trap_ratio: f64,
    /// Write the seen phrasings the orchestrator's rules read to this file,
    /// then stop.
    #[arg(long)]
    emit_templates: Option<PathBuf>,
    /// Write `--names` invented names drawn from `--seed`, one per line, to
    /// this file, then stop: the delexicalised corpus takes its names here.
    #[arg(long)]
    emit_names: Option<PathBuf>,
    /// How many names `--emit-names` writes.
    #[arg(long, default_value_t = 200_000)]
    names: usize,
}

fn parse_ratio(text: &str) -> Result<f64, String> {
    let r: f64 = text.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&r) {
        Ok(r)
    } else {
        Err("must be within [0, 1]".into())
    }
}

/// Writes a side file instead of worlds.
fn emit(path: &PathBuf, text: &str) -> ExitCode {
    match std::fs::write(path, text) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Some(path) = &args.emit_templates {
        return emit(path, &docta_worldgen::export::seen_templates_json());
    }
    if let Some(path) = &args.emit_names {
        let mut text = docta_worldgen::pool::pool(args.seed, args.names).join("\n");
        text.push('\n');
        return emit(path, &text);
    }
    let cfg = Config {
        seed: args.seed,
        worlds: args.worlds,
        ratios: args.split_ratio,
        options: Options {
            unanswerable_ratio: args.unanswerable_ratio,
            max_hops: args.max_hops,
            trap_ratio: args.trap_ratio,
        },
    };
    match write(&cfg, &args.out) {
        Ok(counts) => {
            for (split, c) in counts {
                eprintln!(
                    "{}: {} worlds, {} questions, {} with a trap",
                    split.as_str(),
                    c.worlds,
                    c.questions,
                    c.traps
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
