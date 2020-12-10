import logging as log
import numpy as np
from gomoku import Gomoku
from MCTS import MCTS

class NeuralMCTSPlayer():
    def __init__(self, game, n_mcts_per_step):
        self.mcts = MCTS(game)
        self.n_mcts_per_step = n_mcts_per_step

    def get_move(self, standardBoard):
        self.mcts.reset()
        pi = self.mcts.getActionProb(standardBoard, self.n_mcts_per_step)
        move = np.unravel_index(np.argmax(pi), pi.shape)
        assert(np.sum(standardBoard[:, move[0], move[1]]) == 0)
        return move


if __name__ == "__main__":
    # from gamegui import GameGUI, GUIPlayer
    #
    # g = Gomoku(11, True)
    # p1 = GUIPlayer(1, g.gui)
    # # p2 = NeuralMCTSPlayer(g, 100)
    # p2 = GUIPlayer(2, g.gui)
    # print('start GUI game, close window to exit.')
    # g.play(p1, p2)
    #
    # g.gui.draw_result(g.result)
    # g.gui.wait_to_exit()
    import numpy as np

    # my_array = np.array([[1,1], [1,1]])
    # my_dict = {}
    # my_dict[my_array.tobytes()] = "a"
    #
    # array_2 = np.array([[1,1], [1,1]])
    # my_dict[array_2.tobytes()] = "b"
    my_dict = {(1,2):"a", (2, 3):"B"}

    if (2,3) in my_dict:
        print("got it")

    print(my_dict[(2,3)])

