"""
Tic Tac Toe game implementation.

Tic Tac Toe is a two-player game where players take turns marking a 3x3 grid with their respective symbols,
usually 'X' and 'O'. The objective is to be the first player to get three of their symbols in a row, column,
 or diagonal.
"""
import numpy as np
from collections import namedtuple

# Define a named tuple to store the game result
Result = namedtuple("Result", ["symbol", "is_win", "is_draw", "game_over"])


class TicTacToe:
    """Tic Tac Toe game class."""

    def __init__(self, player_1: str = 'X', player_2: str = 'O', initial_player: str = 'X', n: int = 3):
        """Initialize the game."""
        # Create constants
        self.NUM_ROWS = self.NUM_COLS = n
        self.GRID_SIZE = (self.NUM_ROWS, self.NUM_COLS)
        self.TOTAL_POSSIBLE_MOVES = self.NUM_ROWS * self.NUM_COLS
        self.TOTAL_POSSIBLE_COMBINATIONS = self.NUM_ROWS ** self.TOTAL_POSSIBLE_MOVES

        # Maintain state of the game
        self.players = [player_1, player_2]
        self.can_play = True
        self.board = np.full(self.GRID_SIZE, '', dtype=str)
        self.current_player = self.players.index(initial_player)
    
    def reset(self, initial_player: str = 'X'):
        """Reset the game."""
        self.can_play = True
        self.board = np.full(self.GRID_SIZE, '', dtype=str)
        self.current_player = self.players.index(initial_player)
        
    def __str__(self):
        """Return a string representation of the board."""
        return '\n'.join([' '.join(cell if cell != '' else '.' for cell in row) for row in self.board])

    def play(self, index: int) -> Result:
        """Play a move on the board."""
        if not self.can_play:
            raise RuntimeError(f'Game is over. Please reset the game.')
        if not (0 <= index <= self.TOTAL_POSSIBLE_MOVES - 1):
            raise ValueError(f'Invalid move. Index should be between 0 and {self.TOTAL_POSSIBLE_MOVES - 1}.')

        # Convert index to row and col
        # Ensure the cell is empty
        row, col = divmod(index, self.NUM_COLS)
        if not self.board[row, col] == '':
            raise ValueError("Invalid move. Cell already occupied.")

        # Make the move
        current_symbol = self.players[self.current_player]
        self.board[row, col] = current_symbol

        # Check game status
        is_win, is_draw = self.get_game_status(current_symbol)

        # Update game state and return result
        game_over = is_win or is_draw
        self.can_play = not game_over
        if not game_over:
            self.update_current_player()
        return Result(current_symbol, is_win, is_draw, game_over)
    
    def update_current_player(self):
        """Update the current player."""
        self.current_player = 1 - self.current_player
    
    def get_game_status(self, player_symbol: str):
        """Check if the game is won or drawn."""
        def is_win():
            # Check rows and columns
            for i in range(self.NUM_ROWS):
                if np.all(self.board[i, :] == player_symbol) or np.all(self.board[:, i] == player_symbol):
                    return True
            # Check diagonals
            if np.all(np.diag(self.board) == player_symbol) or np.all(np.diag(np.fliplr(self.board)) == player_symbol):
                return True
            return False

        def is_draw():
            return np.all(self.board != '')

        return is_win(), is_draw()
