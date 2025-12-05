# Generalization of TicTacToe for (potentially) bigger grid and (maybe) longer string requirement for victory
from itertools import product
from typing import Tuple, List
import random
from copy import deepcopy


alphabet = ''.join([chr(c) for c in range(ord('a'), ord('z')+1)])
max_grid_size = len(alphabet)
space_char = '░'


def convert_move_to_coords(move: str):
    """
    Converts a string like "c5" to row = 4, col = 2
    """
    orig_move = move[:]
    move = move.strip().replace(' ', '').lower()
    if len(move) != 2:
        print(f'{orig_move} is not a valid move notation! (1)')
        return None
    move_col, move_row = move
    if not (move_row.isdigit() and move_col.lower() in alphabet):
        print(f'{orig_move} is not a valid move notation! (2)')
        return None
    move_row = int(move_row)-1
    move_col = alphabet.index(move_col)
    return (move_row, move_col)


def convert_coords_to_move(coords: Tuple[int, int]):
    """
    Converts row = 4, col = 2 to a string like "c5"
    """
    c_row, c_col = coords
    c_col = alphabet[c_col]
    return c_col.upper() + str(c_row+1)


def convert_coords_list_to_moves(coords_list: List):
    """
    Converts a list of (1, 2) to a comma-separated string
    """
    return ', '.join(sorted(convert_coords_to_move(c) for c in coords_list))


class TicTacToeNxN:
    N = None                # Size of the grid
    grid = None             # Grid: '' is empty, 'X' is x, 'O' is o. This is a list of rows: thus, coordinates are (row, col), or (y, x)
    victory_len = None      # Length of tokens required for victory
    last_move = None        # Coordinates of the last placed token as pair (y, x); used to check if there is victory condition
    tokens_placed = None    # Current number of tokens on the board; used to check if the game is over
    possible_moves = None   # List of possible moves as pairs (y, x)
    current_player = None   # Either 'x' or 'o' (or upper() version thereof)

    # alphabet = ''.join([chr(c) for c in range(ord('a'), ord('z')+1)])
    # max_grid_size = len(alphabet)

    # space_char = '░'


    def __init__(self, N: int=3, victory_len: int=3, starting_player: str='x'):
        """
        Creates TicTacToe game in NxN grid with victory condition of forming a line of victory_len tokens
        """
        if N > max_grid_size:
            print(f'WARNING! Grid size too big; capping to {max_grid_size}')
        self.N = min(N, max_grid_size)

        self.grid = [['' for i in range(self.N)] for j in range(self.N)]
        self.victory_len = victory_len
        self.tokens_placed = 0
        self.possible_moves = set(product(range(self.N), range(self.N)))
        self.current_player = starting_player


    def print_grid(self):
        """
        Simple printing of the grid
        """
        #  ABCDE
        # 1░░░x░
        # 2░x░░░
        # 3░░░░░
        # 4░░░o░
        # 5░o░░░
        print(' ' + ''.join(alphabet[:self.N]).upper())
        for i, row in enumerate(self.grid):
            print(str(i+1) + ''.join(space_char if cell in ('', ' ') else cell.upper() for cell in row))
    

    def is_move_possible(self, move: Tuple[int, int]):
        """
        Check if move is among possible moves
        """
        return (move in self.possible_moves)


    def apply_move(self, move: Tuple[int, int], player=None):
        """
        Apply move (expressed in coords) to the current state of the game
        """
        if not self.is_move_possible(move):
            return None
        move_row, move_col = move
        # 1. Place the current player's token at coords
        if self.grid[move_row][move_col] != '':
            print(f'Somehow the cell at {move_row}, {move_col} is not empty')
            return None
        current_player = self.current_player if player is None else player
        self.grid[move_row][move_col] = current_player
        # 2. Exclude this move from list of possible moves
        self.possible_moves.remove(move)
        # 3. Mark last move
        self.last_move = move[:]
        # 4. Increment tokens_placed
        self.tokens_placed += 1

    
    def get_cell_at_coords(self, coords: Tuple[int, int]):
        """
        Self-descriptive
        """
        c_row, c_col = coords
        if not (0 <= c_row < self.N and 0 <= c_col < self.N):
            return None
        return self.grid[c_row][c_col]


    def check_victory(self, player=None):
        """
        Check if the grid satisfy the victory condition for the (current) player
        """
        current_player = self.current_player if player is None else player
        # print(f'\tChecking victory conditions for: {current_player}')
        victory_str = ''.join([current_player for _ in range(self.victory_len)])
        # 1. Horizontal
        if any([victory_str in ''.join(row) for row in self.grid]):
            # print('\tHorizontal line!')
            return True
        # 2. Vertical
        grid_T = zip(*self.grid)
        if any([victory_str in ''.join(col) for col in grid_T]):
            # print('\tVertical line!')
            return True
        # 3. Diagonal
        # 3.1. \
        # print(list(range(self.N - self.victory_len + 1)))
        # print(list(range(self.N - self.victory_len + 1)))
        for r in range(self.N - self.victory_len + 1):
            for c in range(self.N - self.victory_len + 1):
                if self.get_cell_at_coords((r, c)) == current_player:
                    # Check if there is a diagonal of needed length (of 1 less length, starting from this cell)
                    # cells = [self.get_cell_at_coords((r+i, c+i)) for i in range(1, self.victory_len)]
                    # print(f'Cells for ╲: {cells}')
                    has_diag = all([self.get_cell_at_coords((r+i, c+i)) == current_player for i in range(1, self.victory_len)])
                    if has_diag:
                        # print(f'\tDiagonal line! (╲) From cell: {convert_coords_to_move((r, c))}')
                        return True
        # 3.2. /
        # print(list(range(self.N - self.victory_len + 1)))
        # print(list(range(self.N-1, self.victory_len-2, -1)))
        for r in range(self.N - self.victory_len + 1):
            for c in range(self.N-1, self.victory_len-2, -1):
                if self.get_cell_at_coords((r, c)) == current_player:
                    # Check if there is a diagonal of needed length (of 1 less length, starting from this cell)
                    # cells = [self.get_cell_at_coords((r+i, c-i)) for i in range(1, self.victory_len)]
                    # print(f'Cells for ╱: {cells}')
                    has_diag = all([self.get_cell_at_coords((r+i, c-i)) == current_player for i in range(1, self.victory_len)])
                    if has_diag:
                        # print(f'\tDiagonal line! (╱) From cell: {convert_coords_to_move((r, c))}')
                        return True
        # print(f'No victory line')
        return False


    def check_game_end_cond(self):
        """
        Check if any game end conditions are present
        """
        # 1. Check victory conditions for current player
        for player in 'xo':
            if self.check_victory(player=player):
                return f'{player.upper()}_WON'
        # 2. Check if there are no more possible moves
        if self.tokens_placed == self.N * self.N:
            return 'NO_POSSIBLE_MOVES'
        if not self.possible_moves:
            return 'NO_POSSIBLE_MOVES'
        
        return None
    

    def switch_player(self):
        """
        Switch player
        """
        if self.current_player == 'x':
            self.current_player = 'o'
        else:
            self.current_player = 'x'
    

    def play_vs_human(self):
        """
        Basic loop for playing against a fellow scientist
        """
        game_cond = None
        while game_cond == None:
            self.print_grid()
            # 1. Request input
            move = input(f'\nYour move, player {self.current_player}: ')
            coords = convert_move_to_coords(move)
            
            # 2. Move available?
            if not self.is_move_possible(coords):
                print(f'{move} is not possible!')
                continue
            # 3. Do the move
            self.apply_move(coords)
            # 4. Victory?
            game_cond = self.check_game_end_cond()
            # 5. Switch player
            self.switch_player()
        if 'WON' in game_cond:
            print(f'Congratulations! {game_cond}')
        else:
            print(f'Bummer... {game_cond}')
        print('Here`s the final view of the game:')
        self.print_grid()

    
    def play_vs_random(self):
        """
        Basic loop for playing against randomly selected moves
        ------
        verbose: print possible moves and which can lead to what outcome;
        debug: print information in best_result's recursive calls
        """
        game_cond = None
        while game_cond == None:
            self.print_grid()
            # 1. Request input
            if self.current_player == 'x':
                move = input('\nYour move: ')
                coords = convert_move_to_coords(move)
            else:
                coords = random.choice(list(self.possible_moves))
            # 2. Move available?
            if not self.is_move_possible(coords):
                print(f'{move} is not possible!')
                continue
            # 3. Do the move
            self.apply_move(coords)
            # 4. Victory?
            game_cond = self.check_game_end_cond()
            # 5. Switch player
            self.switch_player()
        if 'WON' in game_cond:
            print(f'Congratulations! {game_cond}')
        else:
            print(f'Bummer... {game_cond}')
        print('Here`s the final view of the game:')
        self.print_grid()


    def random_play(self, starting_player='o'):
        """
        Fully random play until some game end condition arises
        """
        game_cond = None
        player = starting_player

        if not len(list(self.possible_moves)):
            # No more possible moves
            game_cond = self.check_game_end_cond()
            return game_cond

        while game_cond == None:
            # 1. Select random move
            coords = random.choice(list(self.possible_moves))
            self.apply_move(coords, player)
            # 2. Check game condition
            game_cond = self.check_game_end_cond()
            # 3. Switch the player
            player = 'x' if player == 'o' else 'o'
        return game_cond
    

    def play_vs_agent(self, agent, verbose=True, debug=False):
        """
        Play against given agent
        """
        game_cond = None
        while game_cond == None:
            self.print_grid()
            # 1. Request input
            if self.current_player == 'x':
                move = input('\nYour move: ')
                coords = convert_move_to_coords(move)
            else:
                coords = agent.select_move(self, self.current_player, verbose=verbose, debug=debug)
            # 2. Move available?
            if not self.is_move_possible(coords):
                print(f'{move} is not possible!')
                continue
            # 3. Do the move
            self.apply_move(coords)
            # 4. Victory?
            game_cond = self.check_game_end_cond()
            # 5. Switch player
            self.switch_player()
        if 'WON' in game_cond:
            print(f'Congratulations! {game_cond}')
        else:
            print(f'Bummer... {game_cond}. A draw!')
        print('Here`s the final view of the game:')
        self.print_grid()


    def agent_vs_agent(self, x_player, o_player, verbose=False, debug=False):
        """
        Simulate a game between two agents: x_player and o_player.
        Uses their select_move methods to play until the game ends.
        """
        game_cond = None
        while game_cond is None:
            if verbose:
                self.print_grid()
            if self.current_player == 'x':
                coords = x_player.select_move(self, self.current_player, verbose=verbose, debug=debug)
            else:
                coords = o_player.select_move(self, self.current_player, verbose=verbose, debug=debug)
            if coords is None:
                return self.check_game_end_cond()
            if not self.is_move_possible(coords):
                raise ValueError(f"Agent selected invalid move: {coords}")
            self.apply_move(coords)
            game_cond = self.check_game_end_cond()
            self.switch_player()
        if verbose:
            print(f'Final game state:')
            self.print_grid()
            if 'WON' in game_cond:
                print(f'Congratulations! {game_cond}')
            else:
                print(f'Bummer... {game_cond}. A draw!')
        return game_cond
    

    ######################################
    ######################################
    # Functions for illustration only
    ######################################
    ######################################
    def find_winning_move(self, game_state, next_player):
        """
        Find a move that immediately wins the game for next_player
        """
        for candidate_move in game_state.possible_moves:
            state_copy = deepcopy(game_state)
            state_copy.apply_move(candidate_move)
            game_end_cond = state_copy.check_game_end_cond()
            if game_end_cond and 'WON' in game_end_cond and game_end_cond.split('_')[0].lower() == next_player:
                return candidate_move
        return None
    

    def eliminate_losing_moves(self, game_state, next_player):
        """
        Avoid giving the opponent a winning move
        """
        opponent = 'x' if next_player == 'o' else 'o'
        possible_moves = []
        for candidate_move in game_state.possible_moves:
            state_copy = deepcopy(game_state)
            state_copy.apply_move(candidate_move)
            opponent_winning_move = self.find_winning_move(state_copy, opponent)
            if opponent_winning_move is None:
                possible_moves.append(candidate_move)
        return possible_moves
    

    def find_two_step_win(self, game_state, next_player):
        """
        Find a two-move sequence that guarantees a win
        """
        opponent = 'x' if next_player == 'o' else 'o'
        for candidate_move in game_state.possible_moves:
            state_copy = deepcopy(game_state)
            state_copy.apply_move(candidate_move)
            good_responses = self.eliminate_losing_moves(state_copy, opponent)
            if not good_responses:
                return candidate_move
        return None
        
    
    ######################################
    ######################################


    def __str__(self):
        """
        Info on the game
        """
        s = f'{self.N = }, {self.victory_len = }\n'
        s += f'{self.tokens_placed = }\n'
        s += f'{self.current_player = }\n'
        s += f'{alphabet = }\n'
        s += f'{max_grid_size = }\n'
        if len(self.possible_moves) < 15:
            s += f'{self.possible_moves = }'
        else:
            s += f'{len(self.possible_moves) = }'
        return s


if __name__ == '__main__':
    ttt5 = TicTacToeNxN(N=5, victory_len=3, starting_player='x')
    
    ttt5.find_winning_move(ttt5, 'x')
    ttt5.eliminate_losing_moves(ttt5, 'x')
    ttt5.find_two_step_win(ttt5, 'x')

    ttt5.play_vs_random()


