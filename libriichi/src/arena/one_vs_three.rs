use super::game::{BatchGame, Index};
use super::result::GameResult;
use crate::agent::{AkochanAgent, BatchAgent, HumanAgent, new_py_agent};
use std::fs::{self, File};
use std::io;
use std::iter;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use flate2::Compression;
use flate2::read::GzEncoder;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Clone, Default)]
pub struct OneVsThree {
    pub disable_progress_bar: bool,
    pub log_dir: Option<String>,
}

#[pymethods]
impl OneVsThree {
    #[new]
    #[pyo3(signature = (*, disable_progress_bar=false, log_dir=None))]
    const fn new(disable_progress_bar: bool, log_dir: Option<String>) -> Self {
        Self {
            disable_progress_bar,
            log_dir,
        }
    }

    /// Returns the total score of the challenger (sum of per-game net delta
    /// divided by 1000, floored per game).
    pub fn py_vs_py(
        &self,
        challenger: PyObject,
        champion: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<i64> {
        // `allow_threads` is required, otherwise it will block python GC to
        // run, leading to memory leaks, since this function is doing long
        // tasks.
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| new_py_agent(challenger, player_ids),
                |player_ids| new_py_agent(champion, player_ids),
                seed_start,
                seed_count,
            )?;

            let total_score: i64 = results
                .iter()
                .enumerate()
                .map(|(i, result)| (result.scores[i % 4] - 25000).div_euclid(1000) as i64)
                .sum();
            Ok(total_score)
        })
    }

    /// Returns the rankings of the challenger (akochan in this case).
    pub fn ako_vs_py(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                |player_ids| new_py_agent(engine, player_ids),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];
            for (i, result) in results.iter().enumerate() {
                let rank = result.rankings().rank_by_player[i % 4];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }

    /// Plays `seed_count` single hanchans with the human fixed at seat 0 (East)
    /// against the given engine at seats 1/2/3. Returns the per-game net score
    /// deltas (final score - 25000) of each player.
    pub fn human_vs_py(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<Vec<[i32; 4]>> {
        py.allow_threads(move || {
            let results = self.run_batch_with_layout(
                |player_ids| HumanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                |player_ids| new_py_agent(engine, player_ids),
                seed_start,
                seed_count,
                1,
                &[[0, 1, 1, 1]],
            )?;

            Ok(results
                .iter()
                .map(|r| r.scores.map(|s| s - 25000))
                .collect())
        })
    }

    /// Returns the rankings of the challenger (python agent in this case).
    pub fn py_vs_ako(
        &self,
        engine: PyObject,
        seed_start: (u64, u64),
        seed_count: u64,
        py: Python<'_>,
    ) -> Result<[i32; 4]> {
        py.allow_threads(move || {
            let results = self.run_batch(
                |player_ids| new_py_agent(engine, player_ids),
                |player_ids| AkochanAgent::new_batched(player_ids).map(|a| Box::new(a) as _),
                seed_start,
                seed_count,
            )?;

            let mut rankings = [0; 4];
            for (i, result) in results.iter().enumerate() {
                let rank = result.rankings().rank_by_player[i % 4];
                rankings[rank as usize] += 1;
            }
            Ok(rankings)
        })
    }
}

impl OneVsThree {
    pub fn run_batch<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed_start: (u64, u64),
        seed_count: u64,
    ) -> Result<Vec<GameResult>>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        self.run_batch_with_layout(
            new_challenger_agent,
            new_champion_agent,
            seed_start,
            seed_count,
            4,
            &[
                [0, 1, 1, 1], // split A
                [1, 0, 1, 1], // split B
                [1, 1, 0, 1], // split C
                [1, 1, 1, 0], // split D
            ],
        )
    }

    fn run_batch_with_layout<C, M>(
        &self,
        new_challenger_agent: C,
        new_champion_agent: M,
        seed_start: (u64, u64),
        seed_count: u64,
        games_per_seed: usize,
        agent_idxs_per_seed: &[[usize; 4]],
    ) -> Result<Vec<GameResult>>
    where
        C: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
        M: FnOnce(&[u8]) -> Result<Box<dyn BatchAgent>>,
    {
        if let Some(dir) = &self.log_dir {
            fs::create_dir_all(dir)?;
        }

        let total_games = seed_count as usize * games_per_seed;
        log::info!(
            "seed: [{}, {}) w/ {:#x}, start {} sets, {} hanchans",
            seed_start.0,
            seed_start.0 + seed_count,
            seed_start.1,
            seed_count,
            total_games,
        );

        let seeds: Vec<_> = (seed_start.0..seed_start.0 + seed_count)
            .flat_map(|seed| iter::repeat_n((seed, seed_start.1), games_per_seed))
            .collect();

        let mut challenger_idx = 0usize;
        let mut champion_idx = 0usize;
        let mut challenger_player_ids = Vec::with_capacity(total_games);
        let mut champion_player_ids = Vec::with_capacity(total_games * 3);
        let mut indexes: Vec<[Index; 4]> = Vec::with_capacity(total_games);

        for _ in 0..seed_count as usize {
            for layout in agent_idxs_per_seed {
                let mut game_indexes = [Index::default(); 4];
                for (seat, &agent_idx) in layout.iter().enumerate() {
                    let player_id_idx = if agent_idx == 0 {
                        challenger_player_ids.push(seat as u8);
                        let i = challenger_idx;
                        challenger_idx += 1;
                        i
                    } else {
                        champion_player_ids.push(seat as u8);
                        let i = champion_idx;
                        champion_idx += 1;
                        i
                    };
                    game_indexes[seat] = Index { agent_idx, player_id_idx };
                }
                indexes.push(game_indexes);
            }
        }

        let mut agents = [
            new_challenger_agent(&challenger_player_ids)?,
            new_champion_agent(&champion_player_ids)?,
        ];
        let batch_game = BatchGame::tenhou_hanchan(self.disable_progress_bar);

        let results = batch_game.run(&mut agents, &indexes, &seeds)?;

        if let Some(dir) = &self.log_dir {
            log::info!("dumping game logs");

            let bar = if self.disable_progress_bar {
                ProgressBar::hidden()
            } else {
                ProgressBar::new(total_games as u64)
            };
            const TEMPLATE: &str = "[{elapsed_precise}] [{wide_bar}] {pos}/{len} {percent:>3}%";
            bar.set_style(ProgressStyle::with_template(TEMPLATE)?.progress_chars("#-"));
            bar.enable_steady_tick(Duration::from_millis(150));

            results
                .par_iter()
                .progress_with(bar)
                .enumerate()
                .try_for_each(|(i, game_result)| {
                    let split_name = ["a", "b", "c", "d"][i % games_per_seed];
                    let (seed, key) = game_result.seed;
                    let filename: PathBuf = [dir, &format!("{seed}_{key}_{split_name}.json.gz")]
                        .iter()
                        .collect();

                    let log = game_result.dump_json_log()?;
                    let mut comp = GzEncoder::new(log.as_bytes(), Compression::best());
                    let mut f = File::create(filename)?;
                    io::copy(&mut comp, &mut f)?;

                    anyhow::Ok(())
                })?;
        }

        Ok(results)
    }
}
