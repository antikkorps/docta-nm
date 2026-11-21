use clap::Parser;
use docta_worldgen::corpus::Options;
use docta_worldgen::dataset::{Config, write};
use docta_worldgen::split::Ratios;
use std::path::PathBuf;
use std::process::ExitCode;

/// Generates fictional worlds, their sources and questions as JSONL v1.
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
    /// Share of answerable retrieve questions whose fact gets a trap
    /// (contradiction, false consensus or undecidable).
    #[arg(long, default_value_t = 0.3, value_parser = parse_ratio)]
    trap_ratio: f64,
    /// Write the seen phrasings the orchestrator's rules read to this file,
    /// then stop.
    #[arg(long)]
    emit_templates: Option<PathBuf>,
}

fn parse_ratio(text: &str) -> Result<f64, String> {
    let r: f64 = text.parse().map_err(|e| format!("{e}"))?;
    if (0.0..=1.0).contains(&r) {
        Ok(r)
    } else {
        Err("must be within [0, 1]".into())
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    if let Some(path) = &args.emit_templates {
        return match std::fs::write(path, docta_worldgen::export::seen_templates_json()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        };
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
            for (split, n) in counts {
                eprintln!("{}: {n} worlds", split.as_str());
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
