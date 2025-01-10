"""OpenAI Gym environment for the game Tic Tac Toe."""
from typing import Any, SupportsFloat

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium.core import ObsType, ActType, RenderFrame

from tictactoe import TicTacToe


class TicTacToeEnv(gym.Env):

    version = "v0"
    name = f"TicTacToe-{version}"
    metadata = {'render.modes': ['ascii']}

    def __init__(self, initial_player: str = 'X', n: int = 3):
        super().__init__()
        assert n >= 3, "Board size should be at least 3x3."
        assert initial_player in ['X', 'O'], "Initial player should be either 'X' or 'O'."

        # Save game settings
        self._game_settings = {
            "n": n,
            "player_1": 'X',
            "player_2": 'O',
            "initial_player": initial_player
        }

        # Set the game
        self.game = TicTacToe(**self._game_settings)
        # Define action and observation space
        # action space is the total possible moves i.e., 0 to total_possible_moves - 1
        self.action_space = Discrete(self.game.TOTAL_POSSIBLE_MOVES)
        self.observation_space = Discrete(self.game.TOTAL_POSSIBLE_COMBINATIONS)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.game.reset(self._game_settings['initial_player'])

        obs = self._get_obs()
        info = {
            "player_1": self._game_settings["player_1"],
            "player_2": self._game_settings["player_2"],
            "current_player": self.game.players[self.game.current_player],
        }
        return obs, info

    def step(
        self,
        action: ActType
    ) -> tuple[ObsType, dict[str, float], bool, bool, dict[str, Any]]:
        result = self.game.play(action)

        obs = self._get_obs()
        rewards = {}
        rewards[self.game.current_player_symbol] = 0.
        rewards[self.game.alternate_player_symbol] = 0.

        done = result.game_over
        if done:
            if result.is_win:
                rewards[result.symbol] = 1.
                player_symbols = self.game.players.copy()
                player_symbols.remove(result.symbol)
                other_player = player_symbols[0]
                rewards[other_player] = -1.
            elif result.is_draw:
                rewards[self.game.current_player_symbol] = 0.5
                rewards[self.game.alternate_player_symbol] = 0.5

        info = {
            "player": self.game.current_player_symbol,
            "result": result
        }
        return obs, rewards, done, False, info

    def render(self, mode='ascii') -> RenderFrame | list[RenderFrame] | None:
        if mode != 'ascii':
            raise NotImplementedError()

        board = self.game.board.copy()
        board[board == ''] = '.'
        print('\n'.join([' '.join(row) for row in board]))
        return None

    def _get_obs(self):
        flattened_board = self.game.board.flatten().copy()
        # convert empty locations with '.' and convert to string
        flattened_board[flattened_board == ''] ='.'
        return (''.join(flattened_board)).lower()

    def sample_action(self) -> int:
        available_moves = self.game.available_moves
        if len(available_moves) < 1:
            raise RuntimeError("No available moves. Game is over.")
        return np.random.choice(available_moves).item()
