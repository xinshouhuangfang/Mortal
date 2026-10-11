use crate::{matches_tu8, t, tu8};
use std::cmp::Ordering;
use std::error::Error;
use std::fmt;
use std::str::FromStr;
use std::sync::LazyLock;

use ahash::AHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const MJAI_PAI_STRINGS_LEN: usize = 3 * 9 + 4 + 3 + 3 + 1;
const MJAI_PAI_STRINGS: [&str; MJAI_PAI_STRINGS_LEN] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // s
    "E", "S", "W", "N", "P", "F", "C", // z
    "5mr", "5pr", "5sr", // aka
    "?",   // unknown
];

const MJAI_PAI_STRINGS2: [&str; MJAI_PAI_STRINGS_LEN] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", // m
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", // p
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", // s
    "1z", "2z", "3z", "4z", "5z", "6z", "7z", // z
    "5mr", "5pr", "5sr", // aka
    "?",   // unknown
];

const DISCARD_PRIORITIES: [u8; 38] = [
    6, 5, 4, 3, 2, 3, 4, 5, 6, // m
    6, 5, 4, 3, 2, 3, 4, 5, 6, // p
    6, 5, 4, 3, 2, 3, 4, 5, 6, // s
    7, 7, 7, 7, 7, 7, 7, // z
    1, 1, 1, // aka
    0, // unknown
];

static MJAI_PAI_STRINGS_MAP: LazyLock<AHashMap<&'static str, Tile>> = LazyLock::new(|| {
    MJAI_PAI_STRINGS2
        .iter()
        .chain(MJAI_PAI_STRINGS.iter())
        .enumerate()
        .map(|(id, &s)| {
            let id = if id < MJAI_PAI_STRINGS2.len() {
                id
            } else {
                id - MJAI_PAI_STRINGS2.len()
            };
            (s, Tile::try_from(id).unwrap())
        })
        .collect()
});

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tile(u8);

#[derive(Debug)]
pub enum InvalidTile {
    Number(usize),
    String(String),
}

impl Tile {
    /// # Safety
    /// Calling this method with an out-of-bounds tile ID is undefined behavior.
    #[inline]
    #[must_use]
    pub const fn new_unchecked(id: u8) -> Self {
        Self(id)
    }

    #[inline]
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self.0
    }
    #[inline]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    #[must_use]
    pub const fn is_aka(self) -> bool {
        false
    }

    #[inline]
    #[must_use]
    pub const fn is_jihai(self) -> bool {
        matches_tu8!(self.0, E | S | W | N | P | F | C)
    }

    #[inline]
    #[must_use]
    pub const fn is_yaokyuu(self) -> bool {
        matches_tu8!(
            self.0,
            1m | 9m | 1p | 9p | 1s | 9s | E | S | W | N | P | F | C
        )
    }

    #[inline]
    #[must_use]
    pub const fn is_unknown(self) -> bool {
        self.0 >= tu8!(?)
    }

    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let kind = self.0 / 9;
        let num = self.0 % 9;

        if kind < 3 {
            Self(kind * 9 + (num + 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 1) % 3)
        }
    }

    #[inline]
    #[must_use]
    pub const fn prev(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let kind = self.0 / 9;
        let num = self.0 % 9;
        if kind < 3 {
            Self(kind * 9 + (num + 9 - 1) % 9)
        } else if num < 4 {
            Self(3 * 9 + (num + 4 - 1) % 4)
        } else {
            Self(3 * 9 + 4 + (num - 4 + 3 - 1) % 3)
        }
    }

    #[inline]
    #[must_use]
    pub const fn augment(self) -> Self {
        if self.is_unknown() {
            return self;
        }
        let kind = self.0 / 9;
        let ret = match kind {
            0 => Self(self.0 + 9),
            1 => Self(self.0 - 9),
            _ => self,
        };
        ret
    }

    /// `Ordering::Equal` iff `self == other`
    #[inline]
    #[must_use]
    pub fn cmp_discard_priority(self, other: Self) -> Ordering {
        let l = self.0 as usize;
        let r = other.0 as usize;
        match DISCARD_PRIORITIES[l].cmp(&DISCARD_PRIORITIES[r]) {
            Ordering::Equal => r.cmp(&l),
            o => o,
        }
    }
}

impl Default for Tile {
    fn default() -> Self {
        t!(?)
    }
}

impl TryFrom<u8> for Tile {
    type Error = InvalidTile;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        Self::try_from(v as usize)
    }
}

impl TryFrom<usize> for Tile {
    type Error = InvalidTile;

    fn try_from(v: usize) -> Result<Self, Self::Error> {
        if v >= MJAI_PAI_STRINGS_LEN {
            Err(InvalidTile::Number(v))
        } else {
            Ok(Self(v as u8))
        }
    }
}

impl FromStr for Tile {
    type Err = InvalidTile;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        MJAI_PAI_STRINGS_MAP
            .get(s)
            .copied()
            .ok_or_else(|| InvalidTile::String(s.to_owned()))
    }
}

impl fmt::Debug for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self, f)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(MJAI_PAI_STRINGS[self.0 as usize])
    }
}

impl<'de> Deserialize<'de> for Tile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let tile = String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)?;
        Ok(tile)
    }
}

impl Serialize for Tile {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl fmt::Display for InvalidTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a valid tile: ")?;
        match self {
            Self::Number(n) => fmt::Display::fmt(n, f),
            Self::String(s) => write!(f, "not a valid tile: \"{s}\""),
        }
    }
}

impl Error for InvalidTile {}
