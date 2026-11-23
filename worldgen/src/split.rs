//! Train/val/test/ood assignment, per world and never per question.
//!
//! `ood` (out of distribution) worlds are written with sentence and question
//! phrasings, and chain combinations, that no other split ever sees: success
//! there means the method generalises, not that templates were memorised.
//! `val` worlds are written in phrasings of their own, held out of train and
//! test too, so settings can be chosen for generalisation without ood.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    Train,
    Val,
    Test,
    Ood,
}

impl Split {
    pub const ALL: [Split; 4] = [Split::Train, Split::Val, Split::Test, Split::Ood];

    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
            Split::Ood => "ood",
        }
    }

    /// Whether this split uses the held-out phrasings and chain combinations.
    pub fn held_out(self) -> bool {
        self.wording() == Wording::OodHeld
    }

    /// Which phrasings this split's texts are written in.
    pub fn wording(self) -> Wording {
        match self {
            Split::Train | Split::Test => Wording::Seen,
            Split::Val => Wording::ValHeld,
            Split::Ood => Wording::OodHeld,
        }
    }
}

/// The phrasings a text draws from: train's, val's own, or ood's own. val is
/// written in phrasings train never sees, so settings chosen on val are chosen
/// for generalisation, while ood stays sealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wording {
    Seen,
    ValHeld,
    OodHeld,
}

/// Share of worlds per split; `ood` takes the rest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    pub train: f64,
    pub val: f64,
    pub test: f64,
}

impl Ratios {
    /// Parses "train,val,test,ood", e.g. "0.7,0.1,0.1,0.1".
    pub fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<f64> = text
            .split(',')
            .map(|p| p.trim().parse::<f64>().map_err(|e| format!("{p:?}: {e}")))
            .collect::<Result<_, _>>()?;
        let [train, val, test, ood] = parts[..] else {
            return Err(format!(
                "expected 4 ratios train,val,test,ood, got {}",
                parts.len()
            ));
        };
        if [train, val, test, ood]
            .iter()
            .any(|r| !(0.0..=1.0).contains(r))
        {
            return Err("each ratio must be within [0, 1]".into());
        }
        let sum = train + val + test + ood;
        if (sum - 1.0).abs() > 1e-9 {
            return Err(format!("ratios must sum to 1, got {sum}"));
        }
        Ok(Self { train, val, test })
    }
}

/// Derives the split from the world seed alone, so generating more worlds never
/// moves an existing world to another split. Proportions are approximate.
pub fn assign(seed: u64, ratios: &Ratios) -> Split {
    // World seeds are uniform over 63 bits: map them to [0, 1).
    let u = seed as f64 / (1u64 << 63) as f64;
    let bounds = [
        (ratios.train, Split::Train),
        (ratios.train + ratios.val, Split::Val),
        (ratios.train + ratios.val + ratios.test, Split::Test),
    ];
    bounds
        .iter()
        .find(|(bound, _)| u < *bound)
        .map_or(Split::Ood, |(_, split)| *split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::seed::world_seed;

    const DEFAULT: Ratios = Ratios {
        train: 0.7,
        val: 0.1,
        test: 0.1,
    };

    #[test]
    fn parses_valid_ratios() {
        assert_eq!(Ratios::parse("0.7,0.1,0.1,0.1"), Ok(DEFAULT));
        assert_eq!(Ratios::parse(" 0.5 , 0.5 , 0, 0 ").map(|r| r.val), Ok(0.5));
    }

    #[test]
    fn rejects_invalid_ratios() {
        for bad in [
            "0.8,0.1,0.1",
            "a,b,c,d",
            "0.7,0.1,0.1,0.2",
            "1.1,-0.1,0,0",
            "0.7,0.1,0.1,0.1,0",
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
        let expected = [
            (Split::Train, 0.7),
            (Split::Val, 0.1),
            (Split::Test, 0.1),
            (Split::Ood, 0.1),
        ];
        for (split, ratio) in expected {
            let share = count(split) as f64 / n as f64;
            assert!((share - ratio).abs() < 0.01, "{split:?}: {share}");
        }
    }

    #[test]
    fn zero_ratio_means_empty_split() {
        let no_ood = Ratios {
            train: 0.8,
            val: 0.1,
            test: 0.1,
        };
        assert!((0..5000).all(|i| assign(world_seed(1, i), &no_ood) != Split::Ood));
    }

    #[test]
    fn only_ood_holds_out_surface_forms() {
        assert!(Split::Ood.held_out());
        assert!(!Split::Train.held_out() && !Split::Val.held_out() && !Split::Test.held_out());
    }
}
