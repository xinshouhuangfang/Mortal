use super::item::KawaItem;
use super::{PlayerState};
use crate::array::Simple2DArray;
use crate::consts::{ACTION_SPACE, MAX_VERSION, obs_shape};
use crate::tile::Tile;
use crate::{tuz};

use ndarray::prelude::*;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

const SELF_KAWA_ITEM_CHANNELS: usize = 4;
const KAWA_ITEM_CHANNELS: usize = 8;
const MAX_NUM_TURNS: usize = 17; // aka the actual practical `MAX_TSUMOS_LEFT`

struct ObsEncoderContext<'a> {
    state: &'a PlayerState,
    arr: Simple2DArray<34, f32>,
    mask: Array1<bool>,
    idx: usize,
    at_kan_select: bool,
    version: u32,
}

#[must_use]
struct IntegerEncoder {
    n: usize,
    cap: usize,
    one_hot: bool,
    rescale: bool,
}

impl IntegerEncoder {
    const fn new(n: usize, cap: usize) -> Self {
        Self {
            n,
            cap,
            one_hot: false,
            rescale: false,
        }
    }
    const fn one_hot(mut self, v: bool) -> Self {
        self.one_hot = v;
        self
    }

    fn encode(self, ctx: &mut ObsEncoderContext<'_>) {
        let n = self.n.min(self.cap);
        debug_assert!(self.one_hot || self.rescale);
        if self.one_hot {
            ctx.arr.fill(ctx.idx + n, 1.);
            ctx.idx += self.cap + 1;
        }
        if self.rescale {
            let v = n as f32 / self.cap as f32;
            ctx.arr.fill(ctx.idx, v);
            ctx.idx += 1;
        }
    }
}

impl<'a> ObsEncoderContext<'a> {
    fn new(state: &'a PlayerState, version: u32, at_kan_select: bool) -> Self {
        assert!(version <= MAX_VERSION);
        let shape = obs_shape(version);
        let arr = Simple2DArray::new(shape.0);
        let mask = Array1::default(ACTION_SPACE);
        Self {
            state,
            arr,
            mask,
            idx: 0,
            at_kan_select,
            version,
        }
    }

    fn encode_obs(mut self) -> (Array2<f32>, Array1<bool>) {
        let state = self.state;
        let cans = state.last_cans;

        state
            .tehai
            .iter()
            .enumerate()
            .filter(|&(_, &count)| count > 0)
            .for_each(|(tile_id, &count)| {
                let n = count as usize;
                self.arr.assign_rows(self.idx, tile_id, n, 1.);
            });
        self.idx += 4;

        self.encode_tile_set(state.dora_indicators);

        state.kawa[0]
            .iter()
            .take(6)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        self.idx += (6 - state.kawa[0].len().min(6)) * SELF_KAWA_ITEM_CHANNELS;

        state.kawa[0]
            .iter()
            .rev()
            .take(18)
            .for_each(|kawa_item| self.encode_self_kawa(kawa_item.as_ref()));
        self.idx += (18 - state.kawa[0].len().min(18)) * SELF_KAWA_ITEM_CHANNELS;

        let max_kawa_len = state.kawa.iter().map(|k| k.len()).max().unwrap();
        for (turn, kawa_item) in state.kawa[0].iter().enumerate() {
            if let Some(kawa_item) = kawa_item {
                let sutehai = kawa_item.sutehai;
                let tid = sutehai.tile.as_usize();
                let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                self.arr.assign(self.idx, tid, v);
            }
        }
        self.idx += 1;

        for player_kawa in &state.kawa[1..] {
            player_kawa
                .iter()
                .take(6)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            self.idx += (6 - player_kawa.len().min(6)) * KAWA_ITEM_CHANNELS;

            player_kawa
                .iter()
                .rev()
                .take(18)
                .for_each(|kawa_item| self.encode_kawa(kawa_item.as_ref()));
            self.idx += (18 - player_kawa.len().min(18)) * KAWA_ITEM_CHANNELS;

            for (turn, kawa_item) in player_kawa.iter().enumerate() {
                if let Some(kawa_item) = kawa_item {
                    let sutehai = kawa_item.sutehai;
                    let tid = sutehai.tile.as_usize();
                    let v = (-0.2 * (max_kawa_len - 1 - turn) as f32).exp();
                    self.arr.assign(self.idx, tid, v);
                    if sutehai.is_tedashi {
                        self.arr.assign(self.idx + 1, tid, v);
                    }
                    if sutehai.is_riichi {
                        self.arr.assign(self.idx + 2, tid, v);
                    }
                }
            }
            self.idx += 3;
        }

        let v = state.tiles_left as f32 / 84.;
        self.arr.fill(self.idx, v);
        self.idx += 1;

        for player_kawa_overview in &state.kawa_overview {
            self.encode_tile_set(player_kawa_overview.iter().copied());
        }

        for player_fuuro in &state.fuuro_overview {
            for f in player_fuuro {
                for tile in f {
                    let tile_id = tile.as_usize();
                    let i = (0..4)
                        .find(|&i| self.arr.get(self.idx + i, tile_id) == 0.)
                        .unwrap();
                    self.arr.assign(self.idx + i, tile_id, 1.);
                    // It is not possible to have more than one aka in a fuuro
                    // set, at least in tenhou rule, so we simply use one
                    // channel here.
                    if tile.is_aka() {
                        self.arr.fill(self.idx + 4, 1.);
                    }
                }
                self.idx += 5;
            }
            self.idx += (4 - player_fuuro.len()) * 5;
        }

        for player_ankan in &state.ankan_overview {
            for tile in player_ankan {
                let tile_id = tile.as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }
            self.idx += 1;
        }

        if matches!(self.version, 2 | 3 | 4) {
            for (tid, count) in state.tiles_seen.iter().copied().enumerate() {
                self.arr.assign(self.idx, tid, count as f32 / 4.);
            }
            self.idx += 1;

            for &player_last_tedashi in &state.last_tedashis[1..] {
                if let Some(sutehai) = player_last_tedashi {
                    let tile = sutehai.tile;
                    let tile_id = tile.as_usize();

                    self.arr.assign(self.idx, tile_id, 1.);
                }
                self.idx += 3;
            }
            for &player_riichi_sutehai in &state.riichi_sutehais[1..] {
                if let Some(sutehai) = player_riichi_sutehai {
                    let tile = sutehai.tile;
                    let tile_id = tile.as_usize();

                    self.arr.assign(self.idx, tile_id, 1.);
                }
                self.idx += 3;
            }
        }

        state
            .waits
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c)
            .for_each(|(t, _)| self.arr.assign(self.idx, t, 1.));
        self.idx += 1;

        let n = state.shanten as usize;
        IntegerEncoder::new(n, 6).one_hot(true).encode(&mut self);

        if self.at_kan_select {
            self.arr.fill(self.idx, 1.);
        }
        self.idx += 1;

        if cans.can_pass() {
            let tile = state
                .last_kawa_tile
                .expect("building chi/pon/daiminkan/ron feature without any kawa tile");
            let tile_id = tile.as_usize();

            self.arr.assign(self.idx, tile_id, 1.);
            if state.dora_factor[tile.as_usize()] > 0 {
                self.arr.fill(self.idx + 2, 1.);
            }

            // pass
            if !self.at_kan_select {
                self.mask[ACTION_SPACE - 1] = true;
            } else if cans.can_daiminkan {
                self.mask[tile_id] = true;
            }
        }
        self.idx += 3;

        if cans.can_discard {
            state
                .discard_candidates_aka()
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| {
                    self.arr.assign(self.idx, t, 1.);
                    if !self.at_kan_select {
                        self.mask[t] = true;
                    }
                });

            state
                .keep_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 1, t, 1.));
            state
                .next_shanten_discards
                .iter()
                .enumerate()
                .filter(|&(_, &c)| c)
                .for_each(|(t, _)| self.arr.assign(self.idx + 2, t, 1.));

            if state.shanten <= 1 {
                state
                    .discard_candidates_with_unconditional_tenpai()
                    .iter()
                    .enumerate()
                    .filter(|&(_, &c)| c)
                    .for_each(|(t, _)| self.arr.assign(self.idx + 3, t, 1.));
            }
        }
        self.idx += 5;

        if cans.can_pon {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[41] = true;
            }
        }
        self.idx += 1;

        if cans.can_daiminkan {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[42] = true;
            }
        }
        self.idx += 1;

        if cans.can_ankan {
            for tile in state.ankan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true;
                }
            }
            if !self.at_kan_select {
                self.mask[42] = true;
            }
        }
        self.idx += 1;

        if cans.can_kakan {
            for tile in state.kakan_candidates {
                self.arr.assign(self.idx, tile.as_usize(), 1.);
                if self.at_kan_select {
                    self.mask[tile.as_usize()] = true;
                }
            }
            if !self.at_kan_select {
                self.mask[42] = true;
            }
        }
        self.idx += 1;

        if cans.can_agari() {
            self.arr.fill(self.idx, 1.);
            if !self.at_kan_select {
                self.mask[43] = true;
            }
        }
        self.idx += 1;

        assert_eq!(self.idx, self.arr.rows());
        let arr = self.arr.build();
        debug_assert!(arr.iter().all(|&v| (0. ..=1.).contains(&v)));
        (arr, self.mask)
    }

    fn encode_tile_set<I>(&mut self, tiles: I)
    where
        I: IntoIterator<Item = Tile>,
    {
        let mut counts = [0; 34];
        for tile in tiles {
            let tile_id = tile.as_usize();

            let i = &mut counts[tile_id];
            self.arr.assign(self.idx + *i, tile_id, 1.);
            *i += 1;

            if tile.is_aka() {
                let i = tile.as_usize() - tuz!(5mr);
                self.arr.fill(self.idx + 4 + i, 1.);
            }
        }
        self.idx += 7;
    }

    fn encode_self_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            for kan in k.kan {
                // no aka tiles in the local rules
                let tile_id = kan.as_usize();
                self.arr.assign(self.idx, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.as_usize();
            self.arr.assign(self.idx + 1, tile_id, 1.);
            if sutehai.tile.is_aka() {
                self.arr.fill(self.idx + 2, 1.);
            }
            if sutehai.is_dora {
                self.arr.fill(self.idx + 3, 1.);
            }
        }
        self.idx += SELF_KAWA_ITEM_CHANNELS;
    }

    fn encode_kawa(&mut self, item: Option<&KawaItem>) {
        if let Some(k) = item {
            if let Some(cp) = &k.chi_pon {
                // Aka info of the chi/pon is not encoded in the kawa detail;
                // they are included in fuuro_overview instead.
                //
                // This is one-hot.
                let a = cp.consumed[0].as_usize();
                let b = cp.consumed[1].as_usize();
                let min = a.min(b);
                let max = a.max(b);
                self.arr.assign(self.idx, min, 1.);
                self.arr.assign(self.idx + 1, max, 1.);
            }

            for kan in k.kan {
                let tile_id = kan.as_usize();
                self.arr.assign(self.idx + 2, tile_id, 1.);
            }

            let sutehai = k.sutehai;
            let tile_id = sutehai.tile.as_usize();
            self.arr.assign(self.idx + 3, tile_id, 1.);
            if sutehai.tile.is_aka() {
                self.arr.fill(self.idx + 4, 1.);
            }
            if sutehai.is_dora {
                self.arr.fill(self.idx + 5, 1.);
            }
            if sutehai.is_tedashi {
                self.arr.fill(self.idx + 6, 1.);
            }
            if sutehai.is_riichi {
                self.arr.fill(self.idx + 7, 1.);
            }
        }
        self.idx += KAWA_ITEM_CHANNELS;
    }
}

#[pymethods]
impl PlayerState {
    /// Returns `(obs, mask)`
    #[pyo3(name = "encode_obs")]
    fn encode_obs_py<'py>(
        &self,
        version: u32,
        at_kan_select: bool,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<bool>>) {
        let (obs, mask) = self.encode_obs(version, at_kan_select);
        let obs = PyArray2::from_owned_array(py, obs);
        let mask = PyArray1::from_owned_array(py, mask);
        (obs, mask)
    }
}

impl PlayerState {
    /// Returns `(obs, mask)`
    #[must_use]
    pub fn encode_obs(&self, version: u32, at_kan_select: bool) -> (Array2<f32>, Array1<bool>) {
        ObsEncoderContext::new(self, version, at_kan_select).encode_obs()
    }
}
