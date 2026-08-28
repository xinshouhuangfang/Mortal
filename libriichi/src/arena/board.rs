use super::result::KyokuResult;
use crate::array::Simple2DArray;
use crate::consts::oracle_obs_shape;
use crate::mjai::{Event, EventExt};
use crate::state::PlayerState;
use crate::tile::Tile;
use crate::vec_ops::vec_add_assign;
use crate::{must_tile, t, tu8};
use std::convert::TryInto;
use std::{array, mem};

use anyhow::{Context, Result, bail};
use derivative::Derivative;
use ndarray::prelude::*;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use sha3::{Digest, Sha3_256};

/// The fields are all pub on purpose so the caller will be able to set the
/// yama, doras, scores directly.
///
/// Other than what is mentioned below, everything else is identical to Tenhou's
/// Rule.
///
/// 1. No triple-ron ryukyoku.
/// 2. Tenhou (the yaku) and chihou do not accumulate with other yakus; they are
///    always 1x yakuman.
#[derive(Debug, Default)]
pub struct Board {
    /// Counts from 0
    pub kyoku: u8,
    pub honba: u8,
    /// Does not effect the kyoku seed
    pub kyotaku: u8,
    /// [25000; 4]
    pub scores: [i32; 4],

    pub haipai: [[Tile; 13]; 4],
    /// Goes backward (pop)
    ///
    /// 本地规则：无宝牌（表宝/里宝）、无赤宝牌、无岭上自摸区。
    /// 发牌时每人 13 张，剩余 84 张全部进入 `yama` 作为牌山，杠后也从
    /// `yama` 末尾（pop）补牌，不再有独立的 rinshan / dora / ura 栈。
    pub yama: Vec<Tile>,
}

#[derive(Derivative)]
#[derivative(Default)]
pub struct BoardState {
    board: Board,
    // Absolute seat, with the oya of E1 always being 0
    oya: u8,
    player_states: [PlayerState; 4],

    has_hora: bool,
    kyoku_deltas: [i32; 4],

    // 本地规则：牌山 84 张（136 总牌 - 52 手牌）。发牌触发条件见 `step()`。
    #[derivative(Default(value = "84"))]
    tiles_left: u8,
    tsumo_actor: u8,
    kans: u8,

    log: Vec<EventExt>,
}

pub struct AgentContext<'a> {
    pub player_states: &'a [PlayerState; 4],
    pub log: &'a [EventExt],
}

#[derive(Clone, Copy)]
pub enum Poll {
    InGame,
    End,
}

impl Board {
    pub fn init_from_seed(&mut self, game_seed: (u64, u64)) {
        let (nonce, key) = game_seed;
        let kyoku_seed = Sha3_256::new()
            .chain_update(nonce.to_le_bytes())
            .chain_update(key.to_le_bytes())
            .chain_update([self.kyoku, self.honba])
            .finalize()
            .into();
        let mut rng = ChaCha12Rng::from_seed(kyoku_seed);
        let mut seq = UNSHUFFLED;
        seq.shuffle(&mut rng);

        self.haipai = array::from_fn(|i| seq[i * 13..(i + 1) * 13].try_into().unwrap());
        let mut idx = 13 * 4;
        // 本地规则：去掉 rinshan(4) / dora(5) / ura(5) 的分段，发完 52 张手牌后
        // 剩余 84 张全部进入牌山 `yama`。
        self.yama = seq[idx..].to_vec();
        idx += self.yama.len();
        assert_eq!(idx, seq.len());
    }

    pub fn into_state(self) -> BoardState {
        let oya = self.kyoku % 4;

        BoardState {
            board: self,
            oya,
            player_states: array::from_fn(|i| PlayerState::new(i as u8)),
            ..Default::default()
        }
    }
}

impl BoardState {
    /// Returns iff any player on the board can act or the kyoku has ended.
    pub fn poll(&mut self, mut reactions: [EventExt; 4]) -> Result<Poll> {
        loop {
            let poll = self.step(&reactions)?;
            match poll {
                Poll::InGame => {
                    if self.player_states.iter().any(|c| c.last_cans().can_act()) {
                        return Ok(poll);
                    }
                }
                Poll::End => {
                    self.add_log_no_meta(Event::EndKyoku);
                    vec_add_assign(&mut self.board.scores, &self.kyoku_deltas);
                    return Ok(poll);
                }
            };
            reactions = Default::default();
        }
    }

    #[inline]
    pub fn agent_context(&self) -> AgentContext<'_> {
        AgentContext {
            player_states: &self.player_states,
            log: &self.log,
        }
    }

    #[inline]
    pub const fn end(&self) -> KyokuResult {
        KyokuResult {
            kyoku: self.board.kyoku,
            has_hora: self.has_hora,
            kyotaku_left: self.board.kyotaku,
            scores: self.board.scores,
        }
    }

    #[inline]
    pub fn take_log(&mut self) -> Vec<EventExt> {
        mem::take(&mut self.log)
    }

    #[inline]
    fn add_log(&mut self, ev: EventExt) {
        self.log.push(ev);
    }

    #[inline]
    fn add_log_no_meta(&mut self, ev: Event) {
        self.log.push(EventExt::no_meta(ev));
    }

    #[inline]
    fn broadcast(&mut self, ev: &Event) {
        for s in &mut self.player_states {
            s.update(ev).expect("fatal internal bug in BoardState");
        }
    }

    fn haipai(&mut self) -> Result<()> {
        let bakaze = must_tile!(tu8!(E) + self.board.kyoku / 4);
        let start_kyoku = Event::StartKyoku {
            bakaze,
            // 本地规则：无宝牌，`StartKyoku` 的 `dora_marker` 字段（协议必填）
            // 固定传 `1m`，不表示实际宝牌。
            dora_marker: t!(1m),
            kyoku: self.oya + 1,
            honba: self.board.honba,
            kyotaku: self.board.kyotaku,
            oya: self.oya,
            scores: self.board.scores,
            tehais: self.board.haipai,
        };
        self.broadcast(&start_kyoku);
        self.add_log_no_meta(start_kyoku);

        let tile = self
            .board
            .yama
            .pop()
            .context("invalid yama: empty at init")?;
        self.tiles_left -= 1;
        let first_tsumo = Event::Tsumo {
            actor: self.oya,
            pai: tile,
        };
        self.broadcast(&first_tsumo);
        self.add_log_no_meta(first_tsumo);

        Ok(())
    }

    fn exhaustive_ryukyoku(&mut self) {
        let deltas = [0; 4];

        vec_add_assign(&mut self.kyoku_deltas, &deltas);
        let ryukyoku = Event::Ryukyoku {
            deltas: Some(deltas),
        };
        self.add_log_no_meta(ryukyoku);
        // no need to broadcast
    }

    fn handle_hora(
        &mut self,
        single_actor: u8,
        single_target: u8,
        reactions: &[EventExt; 4],
    ) -> Result<()> {
        self.has_hora = true;

        let is_ron = single_actor != single_target;
        self.board.kyotaku = 0; // Unlike honba, kyotaku in self will be cleared

        // 本地规则：无里宝牌，结算时始终传空 `ura_indicators`。由于
        // `agari_points` 对外保留短路（宝牌番恒 0），此参数不影响算分。
        let ura_indicators: Vec<Tile> = vec![];
        let points = reactions
            .iter()
            .map(|ev| match ev.event {
                Event::Hora { actor, .. } => {
                    let point =
                        self.player_states[actor as usize].agari_points(is_ron, &ura_indicators);
                    Some(point).transpose()
                }
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;

        let point = points[single_actor as usize].unwrap();
        let mut deltas = [0; 4];
        deltas.fill(-point.tsumo_ko);
        deltas[single_actor as usize] =
            point.tsumo_total();

        vec_add_assign(&mut self.kyoku_deltas, &deltas);
        let ura_markers = if self.player_states[single_actor as usize].self_riichi_accepted() {
            ura_indicators
        } else {
            Default::default()
        };

        let hora = Event::Hora {
            actor: single_actor,
            target: single_target,
            deltas: Some(deltas),
            ura_markers: Some(ura_markers),
        };
        self.add_log_no_meta(hora);
        // No need to broadcast

        Ok(())
    }

    fn step(&mut self, reactions: &[EventExt; 4]) -> Result<Poll> {
        // 本地规则：开局牌山 84 张，`tiles_left` 初始为 84 时首先发牌。
        if self.tiles_left == 84 {
            self.haipai()?;
            return Ok(Poll::InGame);
        }

        // Validate reactions
        for (actor, ev) in reactions.iter().enumerate() {
            self.player_states[actor]
                .validate_reaction(&ev.event)
                .with_context(|| {
                    format!(
                        "invalid action: {ev:?}\nstate:\n{}",
                        self.player_states[actor].brief_info(),
                    )
                })?;
        }

        let ev = reactions
            .iter()
            .min_by_key(|ev| match ev.event {
                Event::Hora { .. } => 0,
                Event::Daiminkan { .. } | Event::Pon { .. } => 1,
                Event::None => 3,
                _ => 2,
            })
            .unwrap(); // Unwrap is safe because it is proven non-empty

        match ev.event {
            Event::None => {
                if self.tiles_left == 0 {
                    self.exhaustive_ryukyoku();
                    return Ok(Poll::End);
                }
                // 本地规则：无岭上自摸区，摸牌（含杠后补牌）始终从牌山
                // `yama` 末尾 pop。
                let tile = self.board.yama.pop().with_context(|| {
                    format!("tiles left > 0 ({}) but yama is empty", self.tiles_left)
                })?;
                self.tiles_left -= 1;
                let tsumo = Event::Tsumo {
                    actor: self.tsumo_actor,
                    pai: tile,
                };

                self.broadcast(&tsumo);
                self.add_log_no_meta(tsumo);
            }

            Event::Dahai { actor, pai:_, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                self.tsumo_actor = (actor + 1) % 4;
            }

            Event::Pon { .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
            }

            Event::Ankan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());
                self.tsumo_actor = actor;
                self.kans += 1;
            }

            Event::Daiminkan { actor, .. } | Event::Kakan { actor, .. } => {
                self.broadcast(&ev.event);
                self.add_log(ev.clone());

                self.tsumo_actor = actor;
                self.kans += 1;
            }

            Event::Hora { actor, target, .. } => {
                self.handle_hora(actor, target, reactions)?;
                return Ok(Poll::End);
            }

            _ => {
                bail!("unexpected event: {:?}", ev.event);
            }
        };

        Ok(Poll::InGame)
    }

    pub fn encode_oracle_obs(&self, perspective: u8, version: u32) -> Array2<f32> {
        let shape = oracle_obs_shape(version);
        let mut arr = Simple2DArray::<34, f32>::new(shape.0);
        let mut idx = 0;

        self.player_states
            .iter()
            .cycle()
            .skip(perspective as usize + 1)
            .take(3)
            .for_each(|state| {
                state
                    .tehai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &count)| count > 0)
                    .for_each(|(tile_id, &count)| {
                        arr.assign_rows(idx, tile_id, count as usize, 1.);
                    });
                idx += 4;

                state
                    .akas_in_hand()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &has_it)| has_it)
                    .for_each(|(i, _)| arr.fill(idx + i, 1.));
                idx += 3;

                let n = state.shanten() as usize;
                match version {
                    1 => {
                        arr.fill_rows(idx, n, 1.);
                        idx += 6;
                    }
                    2 | 3 | 4 => {
                        arr.fill(idx + n, 1.);
                        idx += 7;

                        let v = n as f32 / 6.;
                        arr.fill(idx, v);
                        idx += 1;
                    }
                    _ => unreachable!(),
                }

                state
                    .waits()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)
                    .for_each(|(t, _)| arr.assign(idx, t, 1.));
                idx += 1;

                if state.at_furiten() {
                    arr.fill(idx, 1.);
                }
                idx += 1;
            });

        let mut encode_tile = |idx: usize, tile: Tile| {
            let tile_id = tile.as_usize();
            arr.assign(idx, tile_id, 1.);
            if tile.is_aka() {
                arr.fill(idx + 1, 1.);
            }
        };

        self.board
            .yama
            .iter()
            .copied()
            .rev()
            .take(self.tiles_left as usize)
            .for_each(|tile| {
                encode_tile(idx, tile);
                idx += 2;
            });
        idx += (69 - self.tiles_left as usize) * 2;

        assert_eq!(idx, shape.0);
        arr.build()
    }
}

#[rustfmt::skip]
const UNSHUFFLED: [Tile; 136] = [
    t!(1m),  t!(1m), t!(1m), t!(1m),
    t!(2m),  t!(2m), t!(2m), t!(2m),
    t!(3m),  t!(3m), t!(3m), t!(3m),
    t!(4m),  t!(4m), t!(4m), t!(4m),
    t!(5m), t!(5m), t!(5m), t!(5m),
    t!(6m),  t!(6m), t!(6m), t!(6m),
    t!(7m),  t!(7m), t!(7m), t!(7m),
    t!(8m),  t!(8m), t!(8m), t!(8m),
    t!(9m),  t!(9m), t!(9m), t!(9m),

    t!(1p),  t!(1p), t!(1p), t!(1p),
    t!(2p),  t!(2p), t!(2p), t!(2p),
    t!(3p),  t!(3p), t!(3p), t!(3p),
    t!(4p),  t!(4p), t!(4p), t!(4p),
    t!(5p), t!(5p), t!(5p), t!(5p),
    t!(6p),  t!(6p), t!(6p), t!(6p),
    t!(7p),  t!(7p), t!(7p), t!(7p),
    t!(8p),  t!(8p), t!(8p), t!(8p),
    t!(9p),  t!(9p), t!(9p), t!(9p),

    t!(1s),  t!(1s), t!(1s), t!(1s),
    t!(2s),  t!(2s), t!(2s), t!(2s),
    t!(3s),  t!(3s), t!(3s), t!(3s),
    t!(4s),  t!(4s), t!(4s), t!(4s),
    t!(5s), t!(5s), t!(5s), t!(5s),
    t!(6s),  t!(6s), t!(6s), t!(6s),
    t!(7s),  t!(7s), t!(7s), t!(7s),
    t!(8s),  t!(8s), t!(8s), t!(8s),
    t!(9s),  t!(9s), t!(9s), t!(9s),

    t!(E), t!(E), t!(E), t!(E),
    t!(S), t!(S), t!(S), t!(S),
    t!(W), t!(W), t!(W), t!(W),
    t!(N), t!(N), t!(N), t!(N),
    t!(P), t!(P), t!(P), t!(P),
    t!(F), t!(F), t!(F), t!(F),
    t!(C), t!(C), t!(C), t!(C),
];
