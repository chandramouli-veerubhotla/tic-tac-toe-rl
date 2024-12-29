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

    def __init__(self, player_1: str = 'X', player_2: str = 'O', initial_player: str = 'X'):
        # Maintain state of the game
        self.can_play = True
        self.players = [player_1, player_2]
        self.board = np.zeros((3, 3), dtype=str)
        self.current_player = self.players.index(initial_player)
    
    def reset(self, initial_player: str = 'X'):
        """Reset the game."""
        self.can_play = True
        self.board = np.zeros((3, 3), dtype=str)
        self.current_player = self.players.index(initial_player)
        
    def __str__(self):
        return str(self.board)

    def play(self, index: int) -> Result:
        """Play a move on the board."""
        assert self.can_play, "Game is over. Please restart the game."
        assert 0 <= index <= 8, "Invalid move. Index should be between 0 and 8."
        # Convert index to row and col
        # Ensure the cell is empty
        row, col = divmod(index, 3)
        if not self.board[row, col] == '':
            raise ValueError("Invalid move. Cell already occupied.")
        
        # Update the cell with player symbol
        # Check if the game finished after the move        
        current_player_symbol = self.players[self.current_player]
        self.board[row, col] = current_player_symbol
        player_won, game_draw = self.get_game_status(current_player_symbol)

        # Update the current player
        # Update game status based on the result
        self.update_current_player()
        self.can_play = not player_won and not game_draw

        return Result(current_player_symbol, player_won, game_draw, not self.can_play)
    
    def update_current_player(self):
        """Update the current player."""
        self.current_player = 1 - self.current_player
    
    def get_game_status(self, player_symbol: str):
        """Check if the game is finished."""
        def is_win() -> bool:
            # Check rows, columns and diagonals
            for i in range(3):
                if np.all(self.board[i, :] == player_symbol):
                    return True
                
                if np.all(self.board[:, i] == player_symbol):
                    return True
                    
                # Check diagonals
                if np.all(np.diag(self.board) == player_symbol):
                    return True
                    
                if np.all(np.diag(np.fliplr(self.board)) == player_symbol):
                    return True
            return False
        
        def is_draw() -> bool:
            return not np.any(self.board == '')
        
        game_won = is_win()
        game_draw = is_draw()

        return game_won, game_draw

       

        
        
    