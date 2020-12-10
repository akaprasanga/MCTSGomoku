import numpy as np
from MCTSBase import TreeNode, MCTSBase, cpuct, EPS
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TreeNodeStructure(TreeNode):

    def __init__(self, board_state, gameend=None):

        super().__init__()
        self._value = 0
        self.board_state = board_state # board state at this node
        self.Qsa = {}                   # action : Qsa_value
        self._n_visits_board = 0
        self.policy_values = None
        # self._n_visits_action = {}
        self._n_visits_action = np.zeros((11, 11), dtype='int32')
        self.selected_action = None
        self.game_end = gameend
        self.valid_action_from_this_state = self.get_valid_moves()
        self.Ps = None

    def find_action(self):
        '''
        Find the action with the highest upper confidence bound to take from the state represented by this MC tree node.
        :return: action as a tuple (x, y), i.e., putting down a piece at location x, y
        '''

        valid_moves = self.valid_action_from_this_state
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in valid_moves:
            if a in self.Qsa:
                ### 1 = self.Ps[s][a]
                u = self.Qsa[a] + cpuct * self.policy_values[a[0], a[1]] * math.sqrt(self._n_visits_board) / (
                        1 + self._n_visits_action[a[0], a[1]])
            else:
                u = cpuct * self.policy_values[a[0], a[1]] * math.sqrt(self._n_visits_board + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        self.selected_action = best_act
        return self.selected_action

    def is_terminal(self):
        '''
        :return: True if this node is a terminal node, False otherwise.
        '''
        if self.game_end == None:
            return False
        else:
            if self.game_end == 1:
                self._value = 1
            else:
                self._value = 0
            return True

    def update(self, v):
        '''
        Update the statistics/counts and values for this node
        :param v: value backup following the action that was selected in the last call of "find_action"
        :return: None
        '''
        a = self.selected_action
        if a in self.Qsa:
            self.Qsa[a] = (self._n_visits_action[a[0], a[1]] * self.Qsa[a] + v) / (self._n_visits_action[a[0], a[1]] + 1)
            self._n_visits_action[a[0], a[1]] += 1
        else:
            self.Qsa[a] = v
            self._n_visits_action[a[0], a[1]] = 1

        self._n_visits_board += 1

    def value(self):
        '''
        :return: the value of the node form the current player's point of view
        '''
        return self._value

    def get_valid_moves(self):
        b = (self.board_state[0, :, :] + self.board_state[1, :, :]) - 1
        ix, jx = np.nonzero(b)
        idx = [i for i in zip(ix, jx)]
        return idx


class MCTS(MCTSBase):
    """
       Monte Carlo Tree Search
       Note the game board will be represented by a numpy array of size [2, board_size[0], board_size[1]]
       """

    def __init__(self, game):
        '''
        Your subclass's constructor must call super().__init__(game)
        :param game: the Gomoku game
        '''
        self.game = game
        self.tree_struc = []
        # self.visit_count = np.zeros(game.board.shape)
        self.model = PolicyValueNet(board_size=(2, 11, 11))

    def reset(self):
        '''
        Clean up the internal states and make the class ready for a new tree search
        :return: None
        '''
        self.tree_struc = []
        # self.visit_count.fill(0)

    def get_visit_count(self, state):
        '''
        Obtain number of visits for each valid (state, a) pairs from this state during the search
        :param state: the state represented by this node
        :return: a board_size[0] X board_size[1] matrix of visit counts. It should have zero at locations corresponding to invalid moves at this state.
        '''

        visit_count = None
        for each_node in self.tree_struc:
            if (each_node.board_state==state).all():
                visit_count = each_node._n_visits_action

        return visit_count

    def get_treenode(self, standardState):
        '''
        Find and return the node corresponding to the standardState in the search tree
        :param standardState: board state
        :return: tree node (None if the state is new, i.e., we need to expand the tree by adding a node corresponding to the state)
        '''
        corresponding_node_in_tree = None
        if len(self.tree_struc) > 0:
            for node in self.tree_struc:
                if (node.board_state==standardState).all():
                    corresponding_node_in_tree = node
        return corresponding_node_in_tree

    def new_tree_node(self, standardState, game_end):
        '''
        Create a new tree node for the search
        :param standardState: board state
        :param game_end: whether game ends after last move, takes 3 values: None-> game not end; 0 -> game ends with a tie; 1-> player who made the last move win
        :return: a new tree node
        '''

        new_node = TreeNodeStructure(standardState, gameend=game_end)
        pi, v = self.model.predict(new_node.board_state)
        new_node._value = v
        new_node.policy_values = pi
        self.tree_struc.append(new_node)

        return new_node


class PolicyValueNet(nn.Module):

    def __init__(self, board_size):
        super(PolicyValueNet, self).__init__()

        self.device = "cpu"
        self.board_size = board_size

        self.fc1 = nn.Linear(in_features=self.board_size[0]*self.board_size[1]*self.board_size[2], out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=16)

        # Two branch for policy and value
        self.policy_head = nn.Linear(in_features=16, out_features=board_size[1]*board_size[2])
        self.value_head = nn.Linear(in_features=16, out_features=1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value_logit = self.value_head(x)

        return F.softmax(policy_logits, dim=1), torch.tanh(value_logit)

    def predict(self, board):

        occupied_positions = board[0, :, :] + board[1, :, :]
        occupied_positions_mask = occupied_positions==1

        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.board_size[0]*self.board_size[1]*self.board_size[2])
        # self.eval()
        with torch.no_grad():
            pi, v = self.forward(board)

        pi = pi.data.cpu().numpy()[0].reshape(11, 11)
        v = v.data.cpu().numpy()[0][0]

        pi[occupied_positions_mask] = 0

        return pi, v

    def train(self, examples):
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        bs = 1
        for epoch in range(10):

            batch_idx = 0

            while batch_idx < int(len(examples) / bs):
                sample_ids = np.random.randint(len(examples), size=bs)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                boards = boards.contiguous()
                target_pis = target_pis.contiguous()
                target_vs = target_vs.contiguous()

                out_pi, out_v = self.model(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_idx += 1


    def loss_pi(self, targets, outputs):
        loss = -(targets * torch.log(outputs)).sum(dim=1)
        return loss.mean()

    def loss_v(self, targets, outputs):
        loss = torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
        return loss

