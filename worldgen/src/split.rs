//! Train/val/test assignment, per world and never per question.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Train,
    Val,
    Test,
}

impl Split {
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    pub train: f64,
    pub val: f64,
}

impl Ratios {
    /// Parses "train,val,test", e.g. "0.8,0.1,0.1".
    pub fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<f64> = text
            .split(',')
            .map(|p| p.trim().parse::<f64>().map_err(|e| format!("{p:?}: {e}")))
            .collect::<Result<_, _>>()?;
        let [train, val, test] = parts[..] else {
            return Err(format!(
                "expected 3 ratios train,val,test, got {}",
                parts.len()
            ));
        };
        if [train, val, test].iter().any(|r| !(0.0..=1.0).contains(r)) {
            return Err("each ratio must be within [0, 1]".into());
        }
        if (train + val + test - 1.0).abs() > 1e-9 {
            return Err(format!("ratios must sum to 1, got {}", train + val + test));
        }
        Ok(Self { train, val })
    }
}

/// Derives the split from the world seed alone, so generating more worlds never
/// moves an existing world to another split. Proportions are approximate.
pub fn assign(seed: u64, ratios: &Ratios) -> Split {
    // World seeds are uniform over 63 bits: map them to [0, 1).
    let u = seed as f64 / (1u64 << 63) as f64;
    if u < ratios.train {
        Split::Train
    } else if u < ratios.train + ratios.val {
        Split::Val
    } else {
        Split::Test
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::world_seed;

    const DEFAULT: Ratios = Ratios {
        train: 0.8,
        val: 0.1,
    };

    #[test]
    fn parses_valid_ratios() {
        assert_eq!(Ratios::parse("0.8,0.1,0.1"), Ok(DEFAULT));
        assert_eq!(Ratios::parse(" 0.5 , 0.5 , 0 ").map(|r| r.val), Ok(0.5));
    }

    #[test]
    fn rejects_invalid_ratios() {
        for bad in [
            "0.8,0.2",
            "a,b,c",
            "0.8,0.1,0.2",
            "1.1,-0.1,0",
            "0.8,0.1,0.1,0",
        ] {
            assert!(Ratios::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn proportions_follow_the_ratios() {
        let n = 20_000;
        let count = |s| {
            (0..n)
                .filter(|&i| assign(world_seed(7, i), &DEFAULT) == s)
                .count()
        };
        for (split, ratio) in [(Split::Train, 0.8), (Split::Val, 0.1), (Split::Test, 0.1)] {
            let share = count(split) as f64 / n as f64;
            assert!((share - ratio).abs() < 0.01, "{split:?}: {share}");
        }
    }

    #[test]
    fn zero_ratio_means_empty_split() {
        let no_val = Ratios {
            train: 0.9,
            val: 0.0,
        };
        assert!((0..5000).all(|i| assign(world_seed(1, i), &no_val) != Split::Val));
    }
}
