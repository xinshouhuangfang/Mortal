use super::{Agent, BatchifiedAgent, InvisibleState};
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;

use anyhow::{Context, Result};
use std::io::{self, BufRead};

/// `HumanAgent` prompts the user for each action via stdin.
pub struct HumanAgent {
    pub player_id: u8,
}

impl HumanAgent {
    pub fn new(player_id: u8) -> Self {
        Self { player_id }
    }

    pub fn new_batched(player_ids: &[u8]) -> Result<BatchifiedAgent<Self>> {
        BatchifiedAgent::new(|id| Ok(Self::new(id)), player_ids)
    }

    fn read_line() -> Result<String> {
        let mut line = String::new();
        let n = io::stdin().lock().read_line(&mut line)?;
        if n == 0 {
            anyhow::bail!("human_agent: input exhausted")
        }
        Ok(line.trim().to_owned())
    }

    /// Parse a discard input: a tile index (0-36 / 1-37), a tile name
    /// (1m..9m/1p..9p/1s..9s/1z..7z), or an honor name (E/S/W/N/P/F/C).
    /// The rules have no aka (red) tiles, so red indices/names map to the
    /// corresponding normal tile via `deaka`.
    fn parse_discard_tile(input: &str) -> Option<Tile> {
        if let Ok(num) = input.parse::<usize>() {
            let idx = if (1..=37).contains(&num) { num - 1 } else { num };
            if idx < 37 {
                return Some(Tile::new_unchecked(idx as u8).deaka());
            }
        }
        if let Ok(tile) = input.parse::<Tile>() {
            if tile.as_usize() < 37 {
                return Some(tile.deaka());
            }
        }
        let z = match input.to_uppercase().as_str() {
            "E" => Some(27),
            "S" => Some(28),
            "W" => Some(29),
            "N" => Some(30),
            "P" => Some(31),
            "F" => Some(32),
            "C" => Some(33),
            _ => None,
        };
        z.map(|id| Tile::new_unchecked(id).deaka())
    }

    /// Parse a user input string and return the corresponding mjai Event.
    /// This is a public function so tests can verify parsing without stdin.
    pub fn parse_input(input: &str, state: &PlayerState, actor: u8) -> Result<Event> {
        let cans = state.last_cans();
        let input_lower = input.to_lowercase();

        Ok(match input_lower.as_str() {
            "riichi" if cans.can_riichi => Event::Reach { actor },
            "tsumo" if cans.can_tsumo_agari => Event::Hora {
                actor,
                target: actor,
                deltas: None,
                ura_markers: None,
            },
            "ron" if cans.can_ron_agari => Event::Hora {
                actor,
                target: cans.target_actor,
                deltas: None,
                ura_markers: None,
            },
            "chi_low" | "chi_l" if cans.can_chi_low => {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                let first = pai.next();
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed: [first, first.next()],
                }
            }
            "chi_mid" | "chi_m" if cans.can_chi_mid => {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed: [pai.prev(), pai.next()],
                }
            }
            "chi_high" | "chi_h" if cans.can_chi_high => {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                let last = pai.prev();
                Event::Chi {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed: [last.prev(), last],
                }
            }
            "pon" if cans.can_pon => {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                Event::Pon {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed: [pai.deaka(); 2],
                }
            }
            "daiminkan" if cans.can_daiminkan => {
                let pai = state.last_kawa_tile().context("no last kawa tile")?;
                Event::Daiminkan {
                    actor,
                    target: cans.target_actor,
                    pai,
                    consumed: [pai.deaka(); 3],
                }
            }
            "kakan" if cans.can_kakan => {
                let candidates = state.kakan_candidates();
                if candidates.is_empty() {
                    anyhow::bail!("no kakan candidates")
                }
                let tile = candidates[0];
                Event::Kakan {
                    actor,
                    pai: tile.deaka(),
                    consumed: [tile.deaka(); 3],
                }
            }
            "ankan" if cans.can_ankan => {
                let candidates = state.ankan_candidates();
                if candidates.is_empty() {
                    anyhow::bail!("no ankan candidates")
                }
                let tile = candidates[0];
                Event::Ankan {
                    actor,
                    consumed: [tile.deaka(); 4],
                }
            }
            "ryukyoku" if cans.can_ryukyoku => Event::Ryukyoku { deltas: None },
            "pass" | "none" if cans.can_pass() => Event::None,
            _ => {
                if cans.can_discard {
                    if let Some(tile) = Self::parse_discard_tile(input) {
                        let deaka = tile.deaka();
                        if state.tehai()[deaka.as_usize()] > 0 {
                            let tsumogiri = state
                                .last_self_tsumo()
                                .is_some_and(|t| t == tile || t == deaka);
                            return Ok(Event::Dahai {
                                actor,
                                pai: tile,
                                tsumogiri,
                            });
                        }
                        anyhow::bail!("你手里没有 {tile}")
                    }
                }
                let discards: Vec<String> = (0..34)
                    .filter(|&t| state.tehai()[t] > 0)
                    .map(|t| Tile::new_unchecked(t as u8).to_string())
                    .collect();
                anyhow::bail!(
                    "invalid input or action not available: {input}. 可打: {}",
                    discards.join(" ")
                )
            }
        })
    }
}

impl Agent for HumanAgent {
    fn name(&self) -> String {
        "human".to_owned()
    }

    fn react(
        &mut self,
        _log: &[EventExt],
        state: &PlayerState,
        _invisible_state: Option<InvisibleState>,
    ) -> Result<EventExt> {
        let cans = state.last_cans();
        let actor = self.player_id;

        eprintln!("\n===== YOUR TURN =====");
        eprintln!("{}", state.brief_info());
        let tehai = state.tehai();
        let aka = state.akas_in_hand();
        eprintln!("手牌: {}", crate::hand::tiles_to_string(&tehai, aka));
        eprintln!("向听: {}", state.shanten());
        if let Some(t) = state.last_self_tsumo() {
            eprintln!("自摸: {t}");
        }
        if let Some(t) = state.last_kawa_tile() {
            eprintln!("last_kawa_tile: {t}");
        }
        eprintln!();

        let mut has_actions = false;

        if cans.can_discard {
            has_actions = true;
            eprintln!("  <tile_name> / <number 0-36> 打牌");
        }
        if cans.can_tsumo_agari {
            has_actions = true;
            eprintln!("  tsumo");
        }
        if cans.can_ron_agari {
            has_actions = true;
            eprintln!("  ron");
        }
        if cans.can_pon        { has_actions = true; eprintln!("  pon"); }
        if cans.can_daiminkan  { has_actions = true; eprintln!("  daiminkan"); }
        if cans.can_kakan      { has_actions = true; eprintln!("  kakan"); }
        if cans.can_ankan      { has_actions = true; eprintln!("  ankan"); }
        if cans.can_pass()     { has_actions = true; eprintln!("  pass / none"); }

        if !has_actions {
            eprintln!("  (no actions available)");
            return Ok(EventExt::no_meta(Event::None));
        }

        loop {
            eprint!("> ");
            let input = Self::read_line()?;
            if input.is_empty() {
                continue;
            }
            match Self::parse_input(&input, state, actor) {
                Ok(ev) => return Ok(EventExt::no_meta(ev)),
                Err(e) => eprintln!("{e}. Try again."),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mjai::Event;
    use crate::state::PlayerState;
    use crate::t;

    #[test]
    fn parse_discard_tile_name() {
        let mut state = PlayerState::new(0);
        state
            .update(&Event::StartKyoku {
                bakaze: t!(E),
                dora_marker: t!(1m),
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais: [
                    [
                        t!(1m), t!(2m), t!(3m), t!(7m), t!(8m), t!(9m),
                        t!(1p), t!(2p), t!(3p), t!(7p), t!(8p), t!(9p), t!(N),
                    ],
                    [t!(?); 13],
                    [t!(?); 13],
                    [t!(?); 13],
                ],
            })
            .unwrap();
        state.update(&Event::Tsumo { actor: 0, pai: t!(E) }).unwrap();
        let ev = HumanAgent::parse_input("1m", &state, 0).unwrap();
        assert!(matches!(ev, Event::Dahai { pai, .. } if pai == t!(1m)));
    }

    #[test]
    fn parse_discard_number() {
        let mut state = PlayerState::new(0);
        state
            .update(&Event::StartKyoku {
                bakaze: t!(E),
                dora_marker: t!(1m),
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais: [
                    [t!(1m), t!(1m), t!(1m), t!(2m), t!(3m), t!(4m), t!(5m), t!(6m), t!(7m), t!(8m), t!(9m), t!(N), t!(N)],
                    [t!(?); 13],
                    [t!(?); 13],
                    [t!(?); 13],
                ],
            })
            .unwrap();
        state.update(&Event::Tsumo { actor: 0, pai: t!(N) }).unwrap();
        let ev = HumanAgent::parse_input("0", &state, 0).unwrap();
        assert!(matches!(ev, Event::Dahai { pai, .. } if pai == t!(1m)));
    }

    #[test]
    fn parse_riichi() {
        let mut state = PlayerState::new(0);
        state
            .update(&Event::StartKyoku {
                bakaze: t!(E),
                dora_marker: t!(3p),
                kyoku: 1,
                honba: 0,
                kyotaku: 0,
                oya: 0,
                scores: [25000; 4],
                tehais: [
                    [
                        t!(1m), t!(2m), t!(3m), t!(7m), t!(8m), t!(9m),
                        t!(1p), t!(2p), t!(3p), t!(7p), t!(8p), t!(9p), t!(C),
                    ],
                    [t!(?); 13],
                    [t!(?); 13],
                    [t!(?); 13],
                ],
            })
            .unwrap();
        state.update(&Event::Tsumo { actor: 0, pai: t!(C) }).unwrap();
        let ev = HumanAgent::parse_input("riichi", &state, 0).unwrap();
        assert!(matches!(ev, Event::Reach { actor: 0 }));
    }

    #[test]
    fn parse_invalid_input() {
        let state = PlayerState::new(0);
        let result = HumanAgent::parse_input("xyz", &state, 0);
        assert!(result.is_err());
    }

    #[test]
    fn batch_construction_compiles() {
        drop(HumanAgent::new_batched(&[0, 1, 2, 3]));
    }
}
