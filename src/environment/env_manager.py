"""ALFWorld environment manager for lifecycle management."""

import os
from pathlib import Path
from typing import Any

import textworld
import textworld.gym
import yaml
from alfworld.agents.environment import get_environment
from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos


# Monkey-patch TextWorld EvalSymbol.derive to fix Python 3.13+ compatibility.
# TextWorld uses locals().update(context["variables"]) then eval(self.expression),
# but locals().update() does NOT affect eval()'s scope in Python 3.13+.
# Fix: pass variables directly to eval() as the locals dict.
try:
    from textworld.envs.pddl.textgen import EvalSymbol, TerminalSymbol

    def _fixed_eval_derive(self, context=None):
        context = context or self.context
        variables = context.get("variables", {})
        value = eval(self.expression, {"__builtins__": {}}, variables)
        return [TerminalSymbol(value)]

    EvalSymbol.derive = _fixed_eval_derive
except ImportError:
    pass


def _resolve_config(config_path: Path) -> dict:
    """Load and resolve ALFWorld config with expanded paths.

    Args:
        config_path: Path to alfworld_config.yaml

    Returns:
        Resolved config dict
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    alfworld_data = os.path.expanduser("~/.cache/alfworld")
    os.environ["ALFWORLD_DATA"] = alfworld_data

    for key in ["data_path", "eval_id_data_path", "eval_ood_data_path"]:
        if key in config.get("dataset", {}):
            config["dataset"][key] = os.path.expandvars(config["dataset"][key])

    if "domain" in config.get("logic", {}):
        path = os.path.expandvars(config["logic"]["domain"])
        if not os.path.isabs(path):
            path = os.path.join(config_path.parent, path)
        config["logic"]["domain"] = path

    if "grammar" in config.get("logic", {}):
        path = os.path.expandvars(config["logic"]["grammar"])
        if not os.path.isabs(path):
            path = os.path.join(config_path.parent, path)
        config["logic"]["grammar"] = path

    return config


def _make_env_from_gamefiles(gamefiles: list[str], config: dict):
    """Create a lightweight TextWorld gym env from a list of game files.

    Bypasses the full AlfredTWEnv init (which scans all 341 game paths).
    Instantaneous compared to the ~1s full init.

    Args:
        gamefiles: List of .tw-pddl game file paths
        config: Resolved ALFWorld config dict

    Returns:
        TextWorld gym environment
    """
    wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
    request_infos = textworld.EnvInfos(
        won=True, admissible_commands=True, extras=["gamefile"]
    )
    training_method = config["general"]["training_method"]
    if training_method == "dagger":
        max_steps = config["dagger"]["training"]["max_nb_steps_per_episode"]
    else:
        max_steps = config["rl"]["training"]["max_nb_steps_per_episode"]

    env_id = textworld.gym.register_games(
        gamefiles,
        request_infos,
        batch_size=1,
        asynchronous=True,
        max_episode_steps=max_steps,
        wrappers=wrappers,
    )
    return textworld.gym.make(env_id)


class EnvManager:
    """Manages ALFWorld environment lifecycle: load, reset, step, task tracking."""

    def __init__(self, split: str = "eval_out_of_distribution", config_path: str | None = None):
        """Initialize environment manager.

        Args:
            split: ALFWorld data split to use (default: eval_out_of_distribution)
            config_path: Path to ALFWorld config file (default: alfworld_config.yaml in project root)
        """
        self.split = split
        self.env = None
        self.current_task_id = None
        self.done = False
        self.score = 0.0
        self.admissible_commands = []

        # Task completion signaling for task_completed tool
        self.task_completed_flag = False
        self.task_completed_success: bool | None = None
        self.task_completed_summary: str | None = None

        # Set config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            self.config_path = project_root / "alfworld_config.yaml"
        else:
            self.config_path = Path(config_path)

        # Cached config + game files (populated by load(), shared via from_gamefiles())
        self._config: dict | None = None
        self._game_files: list[str] | None = None

    def load(self) -> "EnvManager":
        """Load ALFWorld environment (full init — scans all game files).

        Returns:
            Self for chaining.
        """
        self._config = _resolve_config(self.config_path)
        env_type = self._config["env"]["type"]

        env = get_environment(env_type)(self._config, train_eval=self.split)
        self._game_files = env.game_files
        self.env = env.init_env(batch_size=1)

        return self

    @classmethod
    def from_gamefiles(cls, gamefiles: list[str], config: dict) -> "EnvManager":
        """Create an EnvManager for a specific subset of game files.

        Skips the expensive full AlfredTWEnv init. Instantaneous.

        Args:
            gamefiles: List of .tw-pddl game file paths
            config: Pre-resolved ALFWorld config dict

        Returns:
            Ready-to-use EnvManager
        """
        mgr = cls.__new__(cls)
        mgr.split = "eval_out_of_distribution"
        mgr.env = _make_env_from_gamefiles(gamefiles, config)
        mgr.current_task_id = None
        mgr.done = False
        mgr.score = 0.0
        mgr.admissible_commands = []
        mgr.task_completed_flag = False
        mgr.task_completed_success = None
        mgr.task_completed_summary = None
        mgr.config_path = None
        mgr._config = config
        mgr._game_files = gamefiles
        return mgr

    @property
    def game_files(self) -> list[str]:
        """Return the list of game files for this environment."""
        if self._game_files is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._game_files

    @property
    def config(self) -> dict:
        """Return the resolved ALFWorld config."""
        if self._config is None:
            raise RuntimeError("Environment not loaded. Call load() first.")
        return self._config

    def reset(self) -> tuple[str, dict[str, Any]]:
        """Reset environment to a new task.

        Returns:
            (observation_string, info_dict)
        """
        if self.env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # Reset environment - ALFWorld returns batched results
        obs, info = self.env.reset()

        # Extract from batch (always index [0] for batch_size=1)
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        info_dict = info if isinstance(info, dict) else {}

        # Reset state
        self.done = False
        self.score = 0.0
        self.task_completed_flag = False
        self.task_completed_success = None
        self.task_completed_summary = None

        # Store admissible commands if available (un-batch if nested)
        if "admissible_commands" in info_dict:
            ac = info_dict["admissible_commands"]
            if ac and isinstance(ac[0], list):
                ac = ac[0]
            self.admissible_commands = ac

        # Extract task ID from gamefile path
        if "extra.gamefile" in info_dict:
            val = info_dict["extra.gamefile"]
            if isinstance(val, list):
                val = val[0]
            self.current_task_id = val

        return observation, info_dict

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        """Execute action in environment.

        Args:
            action: Action string to execute

        Returns:
            (observation, score, done, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not loaded. Call load() first.")

        # ALFWorld expects batched input
        obs, score, done, info = self.env.step([action])

        # Extract from batch (always index [0] for batch_size=1)
        observation = obs[0] if isinstance(obs, (list, tuple)) else obs
        score_val = score[0] if isinstance(score, (list, tuple)) else score
        done_val = done[0] if isinstance(done, (list, tuple)) else done
        info_dict = info if isinstance(info, dict) else {}

        # Update internal state
        self.done = done_val
        self.score = score_val

        # Update admissible commands if available
        if "admissible_commands" in info_dict:
            self.admissible_commands = info_dict["admissible_commands"]

        return observation, score_val, done_val, info_dict

    def get_task_type(self) -> str:
        """Get current task type.

        ALFWorld task type mapping:
        - pick_and_place_simple -> "pick"
        - look_at_obj_in_light -> "look"
        - pick_clean_then_place_in_recep -> "clean"
        - pick_heat_then_place_in_recep -> "heat"
        - pick_cool_then_place_in_recep -> "cool"
        - pick_two_obj_and_place -> "pick2"

        Returns:
            Task type identifier
        """
        if self.env is None or self.current_task_id is None:
            return "unknown"

        # Extract task type from gamefile path
        # Typical format: .../pick_and_place_simple-...
        task_id = str(self.current_task_id)

        if "pick_and_place_simple" in task_id:
            return "pick"
        elif "look_at_obj_in_light" in task_id:
            return "look"
        elif "pick_clean_then_place_in_recep" in task_id:
            return "clean"
        elif "pick_heat_then_place_in_recep" in task_id:
            return "heat"
        elif "pick_cool_then_place_in_recep" in task_id:
            return "cool"
        elif "pick_two_obj_and_place" in task_id:
            return "pick2"
        else:
            return "unknown"

    def get_task_id(self) -> str:
        """Get current task identifier.

        Returns:
            Task identifier string
        """
        return str(self.current_task_id) if self.current_task_id else "none"
