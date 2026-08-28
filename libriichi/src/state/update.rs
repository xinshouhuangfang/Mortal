use super::PlayerState;
use super::action::ActionCandidate;
use super::item::{ChiPon, KawaItem, Sutehai};
use crate::algo::shanten;
use crate::mjai::Event;
use crate::rankings::Rankings;
use crate::tile::Tile;
use crate::{must_tile, tu8};
use std::cmp::Ordering;
use std::{iter, mem};

use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy)]
pub(super) enum MoveType {
    Tsumo,
    Discard,
    FuuroConsume,
}

impl PlayerState {
    #[inline]
    pub fn update(&mut self, event: &Event) -> Result<ActionCandidate> {
        self.update_with_keep_cans(event, false)
    }

    /// If `keep_cans_on_announce` is true, then ReachAccepted, Dora and Hora
    /// events will keep `self.last_cans`, `self.ankan_candidates` and
    /// `self.kakan_candidates` unchanged from the last update. Currently
    /// setting it to true is only useful in validate_logs.
    pub fn update_with_keep_cans(
        &mut self,
        event: &Event,
        keep_cans_on_announce: bool,
    ) -> Result<ActionCandidate> {
        self.update_inner(event, keep_cans_on_announce)
            .with_context(|| format!("on event {event:?}"))
    }

    fn update_inner(
        &mut self,
        event: &Event,
        keep_cans_on_announce: bool,
    ) -> Result<ActionCandidate> {
        if !keep_cans_on_announce || !event.is_in_game_announce() {
            self.last_cans = ActionCandidate {
                target_actor: event.actor().unwrap_or(self.player_id),
                ..Default::default()
            };
            self.ankan_candidates.clear();
            self.kakan_candidates.clear();
        }

        if self.to_mark_same_cycle_furiten.take().is_some() {
            self.at_furiten = true;
        }
        if self.chankan_chance.take().is_some() {
            self.at_ippatsu = false;
        }

        match *event {
            Event::StartKyoku {
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
            } => self.start_kyoku(
                bakaze,
                dora_marker,
                kyoku,
                honba,
                kyotaku,
                oya,
                scores,
                tehais,
            )?,

            Event::Tsumo { actor, pai } => self.tsumo(actor, pai)?,
            Event::Dahai {
                actor,
                pai,
                tsumogiri,
            } => self.dahai(actor, pai, tsumogiri)?,

            Event::Pon {
                actor,
                target,
                pai,
                consumed,
            } => self.pon(actor, target, pai, consumed)?,

            Event::Daiminkan {
                actor,
                target,
                pai,
                consumed,
            } => self.daiminkan(actor, target, pai, consumed)?,

            Event::Kakan { actor, pai, .. } => self.kakan(actor, pai)?,
            Event::Ankan { actor, consumed } => self.ankan(actor, consumed)?,
            _ => (),
        };

        Ok(self.last_cans)
    }

    #[allow(clippy::too_many_arguments)]
    fn start_kyoku(
        &mut self,
        bakaze: Tile,
        // 本地规则：无宝牌，忽略 `dora_marker`（协议字段，固定为 1m），
        // 不再设置任何 dora 状态。
        _dora_marker: Tile,
        kyoku: u8,
        honba: u8,
        kyotaku: u8,
        oya: u8,
        scores: [i32; 4],
        tehais: [[Tile; 13]; 4],
    ) -> Result<()> {
        self.tehai.fill(0);
        self.waits.fill(false);
        self.dora_factor.fill(0);
        self.tiles_seen.fill(0);
        self.akas_seen.fill(false);
        self.keep_shanten_discards.fill(false);
        self.next_shanten_discards.fill(false);
        self.forbidden_tiles.fill(false);
        self.discarded_tiles.fill(false);

        self.bakaze = bakaze;
        self.honba = honba;
        self.kyotaku = kyotaku;
        self.oya = self.rel(oya) as u8;
        self.jikaze = must_tile!(tu8!(E) + (4 - self.oya) % 4);
        self.kyoku = kyoku - 1;
        self.is_all_last = match self.bakaze.as_u8() {
            tu8!(E) => false,
            tu8!(S) => self.kyoku == 3,
            _ => true,
        };

        self.scores = scores;
        self.scores.rotate_left(self.player_id as usize);

        self.dora_indicators.clear();
        self.doras_owned.fill(0);
        self.doras_seen = 0;
        self.akas_in_hand.fill(false);

        self.ankan_candidates.clear();
        self.kakan_candidates.clear();
        self.chankan_chance = None;

        self.at_ippatsu = false;
        self.at_rinshan = false;
        self.at_furiten = false;
        self.to_mark_same_cycle_furiten = None;

        self.is_menzen = true;
        self.can_w_riichi = true;
        self.is_w_riichi = false;
        self.chis.clear();
        self.pons.clear();
        self.minkans.clear();
        self.ankans.clear();

        self.kans_on_board = 0;
        self.tehai_len_div3 = 4;
        self.has_next_shanten_discard = false;
        // 本地规则：牌山 84 张（与 `BoardState.tiles_left` 同步）。
        self.tiles_left = 84;
        self.at_turn = 0;

        self.kawa.iter_mut().for_each(|k| k.clear());
        self.last_tedashis.fill(None);
        self.kawa_overview.iter_mut().for_each(|k| k.clear());
        self.fuuro_overview.iter_mut().for_each(|k| k.clear());
        self.ankan_overview.iter_mut().for_each(|k| k.clear());
        self.intermediate_kan.clear();
        self.intermediate_chi_pon = None;

        self.riichi_declared.fill(false);
        self.riichi_accepted.fill(false);
        self.riichi_sutehais.fill(None);

        self.last_self_tsumo = None;
        self.last_kawa_tile = None;

        // The updates must be in order and must be placed after all the
        // resets above.
        self.update_rank();
        for &t in &tehais[self.player_id as usize] {
            self.witness_tile(t)?;
            self.move_tile(t, MoveType::Tsumo)?;
        }
        self.update_shanten();
        self.update_waits_and_furiten();
        self.pad_kawa_at_start();

        Ok(())
    }

    fn tsumo(&mut self, actor: u8, pai: Tile) -> Result<()> {
        ensure!(
            self.tiles_left > 0,
            "rule violation: attempt to tsumo from exhausted yama",
        );
        self.tiles_left -= 1;
        if actor != self.player_id {
            return Ok(());
        }
        self.at_turn += 1;

        self.last_cans.can_discard = true;
        self.last_self_tsumo = Some(pai);
        self.witness_tile(pai)?;
        self.move_tile(pai, MoveType::Tsumo)?;

        if !self.riichi_accepted[0] {
            // Does not update shanten
            self.update_shanten_discards();
        }

        if self.waits[pai.as_usize()] {
            self.last_cans.can_tsumo_agari = true;
        }

        // haitei tile cannot be used for kakan or ankan
        if self.tiles_left == 0 {
            return Ok(());
        }

        if true {
            self.tehai
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .for_each(|(tid, &count)| {
                    let tile = must_tile!(tid);
                    if count == 4 {
                        self.last_cans.can_ankan = true;
                        self.ankan_candidates.push(tile);
                    } else if self.pons.contains(&(tid as u8)) {
                        self.last_cans.can_kakan = true;
                        self.kakan_candidates.push(tile);
                    }
                });
        }

        self.last_cans.can_riichi = false;

        Ok(())
    }

    fn dahai(&mut self, actor: u8, pai: Tile, tsumogiri: bool) -> Result<()> {
        let actor_rel = self.rel(actor);
        if actor_rel == 0 {
            self.move_tile(pai, MoveType::Discard)?;
        } else {
            self.witness_tile(pai)?;
        }

        let is_riichi = false;
        let sutehai = Sutehai {
            tile: pai,
            is_dora: false,
            is_tedashi: !tsumogiri,
            is_riichi,
        };
        let kawa_item = KawaItem {
            kan: mem::take(&mut self.intermediate_kan),
            chi_pon: self.intermediate_chi_pon.take(),
            sutehai,
        };
        self.kawa[actor_rel].push(Some(kawa_item));
        self.kawa_overview[actor_rel].push(pai);
        self.last_kawa_tile = Some(pai);

        if !tsumogiri {
            self.last_tedashis[actor_rel] = Some(sutehai);
        }

        if actor_rel == 0 {
            self.forbidden_tiles.fill(false);
            self.at_rinshan = false;
            self.at_ippatsu = false;
            self.can_w_riichi = false;
            self.discarded_tiles[pai.as_usize()] = true;

            // Furiten state will be permanent once riichi is accepted,
            // and of course, the shanten number will be frozen as well,
            // so the calculations are skipped here.
            if true {
                if self.next_shanten_discards[pai.as_usize()] {
                    self.shanten -= 1;
                } else if !self.keep_shanten_discards[pai.as_usize()] {
                    self.update_shanten();
                }
                // Update is here because `self.tiles_seen` has
                // changed so waits may have been changed, also the
                // discarded `pai` might be a winning tile (tsumo agari
                // minogashi) thus furiten status needs to update.
                self.update_waits_and_furiten();
            }

            return Ok(());
        }

        if self.tiles_left == 0 {
            return Ok(());
        }

        self.last_cans.can_pon = self.tehai[pai.as_usize()] >= 2;
        self.last_cans.can_daiminkan =
            self.kans_on_board < 4 && self.tehai[pai.as_usize()] == 3;

        Ok(())
    }

    fn pon(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 2]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        self.fuuro_overview[actor_rel].push(full_set);
        self.intermediate_chi_pon = Some(ChiPon {
            consumed,
            target_tile: pai,
        });
        self.pad_kawa_for_pon_or_daiminkan(actor, target);

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            self.can_w_riichi = false;
            self.at_ippatsu = false;
            return Ok(());
        }

        self.last_cans.can_discard = true;
        self.is_menzen = false;
        self.tehai_len_div3 -= 1;
        // Marked explicitly as `None` to let `Agent` impls set
        // `tsumogiri` to false in the Dahai after Pon
        self.last_self_tsumo = None;

        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.pons.push(pai.as_u8());

        if self.tehai[pai.as_usize()] > 0 {
            self.forbidden_tiles[pai.as_usize()] = true;
        }

        // NOTES: this is 3n+2
        // The shanten can change after pon, for example 122334789 pon 2.
        self.update_shanten();
        self.update_shanten_discards();

        Ok(())
    }

    fn daiminkan(&mut self, actor: u8, target: u8, pai: Tile, consumed: [Tile; 3]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let full_set = consumed.into_iter().chain(iter::once(pai)).collect();
        self.fuuro_overview[actor_rel].push(full_set);
        self.intermediate_kan.push(pai);
        self.pad_kawa_for_pon_or_daiminkan(actor, target);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            self.can_w_riichi = false;
            self.at_ippatsu = false;
            return Ok(());
        }

        self.is_menzen = false;
        self.tehai_len_div3 -= 1;

        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.minkans.push(pai.as_u8());

        // The shanten number and the shape of tenpai (if any) may be
        // changed after a daiminkan.
        //
        // For example: 12223m 456p 12378s + 2m
        self.update_shanten();
        self.update_waits_and_furiten();

        Ok(())
    }

    fn kakan(&mut self, actor: u8, pai: Tile) -> Result<()> {
        let actor_rel = self.rel(actor);
        for fuuro in &mut self.fuuro_overview[actor_rel] {
            if fuuro[0] == pai {
                fuuro.push(pai);
                break;
            }
        }
        self.intermediate_kan.push(pai);
        self.kans_on_board += 1;

        if actor_rel != 0 {
            self.witness_tile(pai)?;
            self.last_kawa_tile = Some(pai); // for getting winning tile in self.agari

            // 槍槓
            if !self.at_furiten && self.waits[pai.as_usize()] {
                self.last_cans.can_ron_agari = true;
                self.to_mark_same_cycle_furiten = Some(());
                self.chankan_chance = Some(());
            } else {
                self.at_ippatsu = false;
            }

            return Ok(());
        }

        self.move_tile(pai, MoveType::FuuroConsume)?;
        self.pons.retain(|&t| t != pai.as_u8());
        self.minkans.push(pai.as_u8());

        // The shanten number and the shape of tenpai (if any) may
        // be changed after an kakan, because the kan'd tile may
        // come from the existing hand.
        if self.next_shanten_discards[pai.as_usize()] {
            self.shanten -= 1;
        } else if !self.keep_shanten_discards[pai.as_usize()] {
            self.update_shanten();
        }
        self.update_waits_and_furiten();

        Ok(())
    }

    fn ankan(&mut self, actor: u8, consumed: [Tile; 4]) -> Result<()> {
        let actor_rel = self.rel(actor);
        let tile = consumed[0];
        self.ankan_overview[actor_rel].push(tile);
        self.intermediate_kan.push(tile);
        self.kans_on_board += 1;

        self.can_w_riichi = false;
        self.at_ippatsu = false;

        if actor_rel != 0 {
            for t in consumed {
                self.witness_tile(t)?;
            }
            return Ok(());
        }

        self.tehai_len_div3 -= 1;
        for t in consumed {
            self.move_tile(t, MoveType::FuuroConsume)?;
        }
        self.ankans.push(tile.as_u8());

        if !self.riichi_accepted[0] {
            // The shanten number and the shape of tenpai (if any) may
            // be changed after an ankan. See the example in daiminkan.
            self.update_shanten();
            self.update_waits_and_furiten();
        }

        Ok(())
    }

    pub(super) const fn rel(&self, actor: u8) -> usize {
        ((actor + 4 - self.player_id) % 4) as usize
    }

    /// Updates `tiles_seen`, `doras_seen` and `akas_seen`.
    ///
    /// Returns an error if we have already witnessed 4 such tiles.
    pub(super) fn witness_tile(&mut self, tile: Tile) -> Result<()> {
        ensure!(
            !tile.is_unknown(),
            "rule violation: attempt to witness an unknown tile",
        );
        let tile_id = tile.as_usize();

        let seen = &mut self.tiles_seen[tile_id];
        ensure!(
            *seen < 4,
            "rule violation: attempt to witness the fifth {tile}",
        );
        *seen += 1;

        Ok(())
    }

    /// Updates `tehai`, `akas_in_hand` and `doras_owned`, but does not update
    /// `tiles_seen` or `doras_seen`.
    ///
    /// Returns an error when trying to discard or consume a tile that the
    /// player doesn't own.
    pub(super) fn move_tile(&mut self, tile: Tile, move_type: MoveType) -> Result<()> {
        let tile_id = tile.as_usize();
        let tehai_tile = &mut self.tehai[tile_id];
        match move_type {
            MoveType::Tsumo => {
                *tehai_tile += 1;
            }
            MoveType::Discard => {
                ensure!(
                    *tehai_tile > 0,
                    "rule violation: attempt to discard {tile} from void",
                );
                *tehai_tile -= 1;
            }
            MoveType::FuuroConsume => {
                ensure!(
                    *tehai_tile > 0,
                    "rule violation: attempt to consume {tile} from void",
                );
                *tehai_tile -= 1;
            }
        }
        Ok(())
    }

    pub(super) fn pad_kawa_for_pon_or_daiminkan(&mut self, abs_actor: u8, abs_target: u8) {
        let mut i = (abs_target + 1) % 4;
        while i != abs_actor {
            let rel = self.rel(i);
            self.kawa[rel].push(None);
            i = (i + 1) % 4;
        }
    }

    pub(super) fn pad_kawa_at_start(&mut self) {
        self.kawa
            .iter_mut()
            .take(self.oya as usize)
            .for_each(|kawa| kawa.push(None));
    }

    /// Can be called at either 3n+1 or 3n+2.
    ///
    /// For 3n+2, the return value of `shanten::calc_all` may be `-1`. We don't
    /// allow `-1` and it will be written as `0` in order for
    /// `_shanten_discards` to be calculated properly.
    pub(super) fn update_shanten(&mut self) {
        self.shanten = shanten::calc_all(&self.tehai, self.tehai_len_div3).max(0);
        debug_assert!(matches!(self.shanten, 0..=6));
    }

    /// Must be called at 3n+2.
    pub(super) fn update_shanten_discards(&mut self) {
        assert!(self.last_cans.can_discard, "tehai is not 3n+2");

        self.next_shanten_discards.fill(false);
        self.keep_shanten_discards.fill(false);
        self.has_next_shanten_discard = false;

        let mut tehai = self.tehai;
        for (tid, &count) in self.tehai.iter().enumerate() {
            // `self.forbidden_tiles[tid]` is not checked here, but it is
            // acceptable because forbidden tiles are always keep-shanten
            // discards, so it won't affect the result of
            // `has_next_shanten_discard`. We will take forbidden_tiles into
            // account when generating discard candidates.
            if count == 0 {
                continue;
            }
            tehai[tid] -= 1;
            let shanten_after = shanten::calc_all(&tehai, self.tehai_len_div3);
            tehai[tid] += 1;
            match shanten_after.cmp(&self.shanten) {
                Ordering::Less => {
                    self.next_shanten_discards[tid] = true;
                    self.has_next_shanten_discard = true;
                }
                Ordering::Equal => {
                    self.keep_shanten_discards[tid] = true;
                }
                _ => (),
            };
        }
    }

    /// Caller must assure current tehai is 3n+1, and `self.shanten` must be up
    /// to date and correct.
    pub(super) fn update_waits_and_furiten(&mut self) {
        assert!(!self.last_cans.can_discard, "tehai is not 3n+1");

        // Reset the furiten flag here for:
        // 1. Clearing same-cycle furiten.
        // 2. The fact that furiten doesn't make sense if we are no longer
        //    tenpai.
        self.at_furiten = false;
        self.waits.fill(false);

        if self.shanten > 0 {
            return;
        }

        for (t, is_wait) in self.waits.iter_mut().enumerate() {
            if self.tehai[t] == 4 {
                // Cannot wait, not even furiten for the 5th tile.
                //
                // However waiting for the 5th tile when all 4 of them are
                // already in the kawa or fuuro is a valid furiten.
                //
                // Note that although [karaten] is not considered as a wait and
                // thus will not be written to the `waits` in this impl anyways,
                // it is still a valid ryukyoku tenpai in our rule spec.
                continue;
            }
            let mut tehai_after = self.tehai;
            tehai_after[t] += 1;

            if shanten::calc_all(&tehai_after, self.tehai_len_div3) == -1 {
                // furiten is not affected by `tiles_seen`
                if self.discarded_tiles[t] {
                    self.at_furiten = true;
                }
                *is_wait = self.tiles_seen[t] < 4;
            }
        }
    }

    pub(super) fn update_rank(&mut self) {
        self.rank = self.get_rank(self.scores);
    }

    pub(super) fn get_rank(&self, mut scores_rel: [i32; 4]) -> u8 {
        let scores_abs = {
            scores_rel.rotate_right(self.player_id as usize);
            scores_rel
        };
        Rankings::new(scores_abs).rank_by_player[self.player_id as usize]
    }
}
