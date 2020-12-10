# AlphaZero:Gomoku (No Trainning involved)

This is a simple implementation of the Monte Carlo Tree Search used in *AlphaZero* for the board game Gomoku (five in a line). For simplicity, Gomoku game with a 11x11 board is considered. 
1. Here "MCTS" which is a subclass of the class MCTSBase. The MCTSBase class already has the overall search process.
2. Another class called TreeNodeStructure is a subclass of the TreeNode class and it completes the implementation of Tree data structure used in the MCTS. 
3. The MCTSs utilizes a basic neural network. Given a state s, the NN estimates the state value of s as well as computes the policy at s (the probability of actions).