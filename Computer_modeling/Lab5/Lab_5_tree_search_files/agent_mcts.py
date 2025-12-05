from typing import List
import math
from copy import deepcopy
import random
from TicTacToeNxN import TicTacToeNxN
from TicTacToeNxN import convert_coords_to_move as c_to_m
from tree_search_helpers import *


class MCTSNode:
    """
    A node in the tree MCTS builds
    """
    game_state = None       # The current state of the game at this node
    parent = None           # The parent MCTSNode that led to this one. None if this is root
    prev_player = None      # Player that just moved
    move = None             # Last move that led directly to this node, made by prev_player

    win_counts = {}         # Statistics about rollouts from this node
    num_rollouts = 0        # How many rollouts have been made from this node
    children = []           # Child nodes of from this node
    unvisited_moves = None  # Available moves from this node

    depth = 0

    def __init__(self, game_state: TicTacToeNxN, parent=None, prev_player='o', move=None, depth=0):
        self.game_state = game_state
        self.parent = parent
        self.prev_player = prev_player
        self.move = move
        self.win_counts = {
            'x': 0,
            'o': 0
        }
        self.num_rollouts = 0
        self.children: List[MCTSNode] = []
        self.unvisited_moves = list(game_state.possible_moves)
        self.depth = depth

    def switch_player(self):
        """
        Switch to other player
        """
        return 'x' if self.prev_player == 'o' else 'o'
    
    def add_random_child(self, debug=False):
        """
        Take one random unvisited move, generate a new MCTSNode for it, add to children list
        """
        # 1. Pop a random possible move
        index = random.randint(0, len(self.unvisited_moves)-1)
        new_move = self.unvisited_moves.pop(index)
        # 2. Since self.prev_player is the previous player, here we switch to the current one
        #    so that he makes the move
        player = self.switch_player()
        # 2. Apply the move
        new_game_state = deepcopy(self.game_state)
        new_game_state.apply_move(new_move, player=player)
        # 3. Create the node for this move
        new_node = MCTSNode(game_state=new_game_state,  # New game state with this move
                            parent=self,                # Parent is the current node
                            prev_player=player,         # This player who just made the move
                            move=new_move,              # The move itself
                            depth=self.depth+1)
        self.children.append(new_node)
        if debug:
            print(' ' * self.depth + f'Adding random child: move={c_to_m(new_move)}')
        return new_node
    
    def record_win(self, winner: str, debug=False):
        """
        Count a new win for winner, increment rollouts count. If no winner - whatever, just increment rollouts count
        """
        winner = winner.lower()
        if debug:
            print(' ' * self.depth + f'Recording a game result: it`s {winner}')
        if winner in 'xo':
            self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self):
        """
        Can this node have children, i.e. are there any more possible moves?
        """
        return len(self.unvisited_moves) > 0
    
    def is_terminal(self):
        """
        Is this node terminal, i.e. does game end here?
        """
        return self.game_state.check_game_end_cond() != None
    
    def winning_frac(self, player: str):
        """
        Statistics of winning in this node for player
        """
        player = player.lower()
        return float(self.win_counts[player]) / float(self.num_rollouts)
    
    def simulate_random_game(self, debug=False):
        """
        Simulate fully random game from node's game state and return its final result
        """
        if self.is_terminal():
            return self.game_state.check_game_end_cond()
        
        # prev_player just moved, now the opponent moves
        opponent = self.switch_player()
        new_game_state = deepcopy(self.game_state)
        result = new_game_state.random_play(starting_player=opponent)
        if debug:
            print(' ' * self.depth + f'Simulated random game with starting player: {opponent}, result: {result}')
        if 'WON' in result:
            return result[0].lower()    # A simpler way to get player token
        else:
            return result


def uct_score(parent_rollouts, child_rollouts, win_frac, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_frac + temperature * exploration


class MCTSAgent:
    """
    A game-playing agent that implements Monte-Carlo Tree Search (MCTS)
    """
    num_rounds = 100
    temperature = 1

    def __init__(self, num_rounds=10, temperature=1):
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_child_uct(self, node: MCTSNode, debug=False):
        """
        Select a child node according to UCT score (relative to current player)
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)

        # This node's prev_player made the move that led to it. Now we look from opponent's perspective
        next_player = node.switch_player()
        if debug:
            print(f'Selecting a child via UCT score for player: {next_player}')

        best_score = -1
        best_child = None
        for child in node.children:
            score = uct_score(
                parent_rollouts=total_rollouts,
                child_rollouts=child.num_rollouts,
                win_frac=child.winning_frac(player=next_player),
                temperature=self.temperature
            )
            if score > best_score:
                best_score = score
                best_child = child
        if debug:
            print(f'Selected child with move {c_to_m(best_child.move)}: UCT score = {best_score:.2f}')
        return best_child

    def select_move(self, game_state: TicTacToeNxN, next_player, verbose=True, debug=False):
        """
        Select a move for a given game_state if now is next_player's turn
        """
        # Root is the current state: from here we decide what to do
        opponent = 'x' if next_player == 'o' else 'o'
        root = MCTSNode(game_state, prev_player=opponent, depth=0)
        if debug:
            print(f'Selecting move from {next_player}`s perspective')

        # Perform rollouts
        for i in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child_uct(node, debug=debug)
            
            if node.can_add_child():
                # Add a new child into the tree
                node = node.add_random_child(debug=debug)

            # Simulate a random game from this node. starting_player is us, we se
            # winner = self.simulate_random_game(node.game_state, starting_player=next_player)
            winner = node.simulate_random_game(debug=debug)

            while node is not None:
                # Propagate the score back up the tree
                node.record_win(winner)
                node = node.parent

        best_move = None
        best_frac = -1.0
        for child in root.children:
            child_frac = child.winning_frac(next_player)
            if child_frac > best_frac:
                best_frac = child_frac
                best_move = child.move
        if debug:
            print(f'Selected move: {c_to_m(best_move)}')
        return best_move
    

if __name__ == '__main__':
    ttt3 = TicTacToeNxN(N=4, victory_len=3, starting_player='x')
    agent = MCTSAgent(num_rounds=125, temperature=1.5)
    
    ttt3.play_vs_agent(agent, verbose=True, debug=False)