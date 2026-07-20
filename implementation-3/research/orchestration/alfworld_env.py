# IMP3 -- single-episode ALFWorld driver, the ALFWorld analogue of WebShop's
# `gym.make("WebAgentTextEnv-v0", ...)` (see episode_runner.py). implementation-1's
# vendored ALFWorld package only ships a Ray-batched env class (`AlfworldEnvs`,
# `env_package/alfworld/envs.py`) -- there's no ready-made single-episode gym
# class the way WebShop's `WebAgentTextEnv` is vendored. Rather than spin up
# Ray for a single, inherently-sequential orchestrator episode, this wraps the
# exact same `base_env.init_env(batch_size=1)` pattern `AlfworldWorker.__init__`
# already uses (env_package/alfworld/envs.py:61-63), called in-process.
#
# `init_env(batch_size=1)` returns a `textworld.gym` env whose `.reset()`/
# `.step([action])` take/return lists of length 1 -- this class unwraps that
# down to plain scalars so it's a drop-in match for episode_runner.py's
# `env.step(action) -> (obs, reward, done, info)` / `env.get_available_actions()`
# / `env.get_instruction_text()` calls, previously satisfied by WebShop's
# `WebAgentTextEnv`.

from typing import Any, Dict, List, Tuple

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment
from agent_system.environments.env_package.alfworld.envs import compute_reward, load_config_file

_TASK_MARKER = "Your task is to: "


def _extract_task(text_obs: str) -> str:
    # mirrors AlfWorldEnvironmentManagerOptions.extract_task, env_manager.py:393-400
    idx = text_obs.find(_TASK_MARKER)
    if idx == -1:
        raise ValueError("Task description not found in text observation.")
    return text_obs[idx + len(_TASK_MARKER):].strip()


class AlfworldSingleEpisodeEnv:
    """One sequential ALFWorld episode, no Ray. `is_train=True` samples from the
    training game pool (config_tw.yaml's `dataset.data_path`); `is_train=False`
    samples from `eval_dataset` (default "eval_in_distribution", matching the
    yaml default `run_alfworld_lite.sh` pins explicitly)."""

    def __init__(self, config_path: str, seed: int = 0, is_train: bool = True, eval_dataset: str = "eval_in_distribution"):
        config = load_config_file(config_path)
        env_type = config["env"]["type"]
        base_env = get_environment(env_type)(config, train_eval="train" if is_train else eval_dataset)
        self._env = base_env.init_env(batch_size=1)
        self._env.seed(seed)
        self._task_description = ""
        self._admissible: List[str] = []

    def reset(self) -> str:
        obs, infos = self._env.reset()
        infos = {k: v[0] for k, v in infos.items()}
        self._task_description = _extract_task(obs[0])
        self._admissible = infos.get("admissible_commands", [])
        return obs[0]

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, scores, dones, infos = self._env.step([action])
        infos = {k: v[0] for k, v in infos.items()}
        self._admissible = infos.get("admissible_commands", []) if not dones[0] else []
        return obs[0], compute_reward(infos, multi_modal=False), bool(dones[0]), infos

    def get_available_actions(self) -> List[str]:
        return self._admissible

    def get_instruction_text(self) -> str:
        return self._task_description
