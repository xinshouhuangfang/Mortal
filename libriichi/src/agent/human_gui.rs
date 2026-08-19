use super::{BatchAgent, HumanAgent, InvisibleState};
use crate::mjai::EventExt;
use crate::state::PlayerState;

use anyhow::{Context, Result, ensure};
use pyo3::intern;
use pyo3::prelude::*;

/// A batch agent for a human player driven by a Python GUI engine.
///
/// On each turn it serializes the raw `PlayerState` into a dict, hands it to
/// the Python engine's `react_state` method, and parses the returned action
/// string (e.g. `"3m"`, `"pon"`, `"riichi"`) into an event using
/// `HumanAgent::parse_input`.
pub struct HumanGuiBatchAgent {
    engine: PyObject,
    name: String,
    player_ids: Vec<u8>,
}

impl HumanGuiBatchAgent {
    pub fn new(engine: PyObject, player_ids: &[u8]) -> Result<Self> {
        ensure!(player_ids.iter().all(|&id| matches!(id, 0..=3)));

        let name = Python::with_gil(|py| {
            let obj = engine.bind_borrowed(py);
            ensure!(
                obj.getattr("react_state")?.is_callable(),
                "missing method react_state",
            );
            Ok(obj.getattr("name")?.extract::<String>()?)
        })?;

        Ok(Self {
            engine,
            name,
            player_ids: player_ids.to_vec(),
        })
    }
}

impl BatchAgent for HumanGuiBatchAgent {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn set_scene(
        &mut self,
        _index: usize,
        _log: &[EventExt],
        _state: &PlayerState,
        _invisible_state: Option<InvisibleState>,
    ) -> Result<()> {
        Ok(())
    }

    fn get_reaction(
        &mut self,
        index: usize,
        _log: &[EventExt],
        state: &PlayerState,
        _invisible_state: Option<InvisibleState>,
    ) -> Result<EventExt> {
        let action: String = Python::with_gil(|py| {
            let d = state.gui_state_dict(py)?;
            let obj = self.engine.bind_borrowed(py);
            let a = obj.call_method1(intern!(py, "react_state"), (d,))?;
            a.extract()
        })?;

        let actor = *self
            .player_ids
            .get(index)
            .context("player id index out of range")?;
        let ev = HumanAgent::parse_input(&action, state, actor)?;
        Ok(EventExt::no_meta(ev))
    }
}
