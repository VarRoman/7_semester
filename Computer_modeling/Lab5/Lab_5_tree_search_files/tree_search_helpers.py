import enum


class GameResult(enum.Enum):
    """
    Possible end game results
    """
    loss = 1
    draw = 2
    win = 3

def reverse_game_result(result):
    """
    If we have result, what does our opponent have? And vice versa
    """
    if result == GameResult.win:
        # One wins - other loses
        return GameResult.loss
    if result == GameResult.loss:
        # One loses - other wins
        return GameResult.win
    # Draw is common
    return GameResult.draw

