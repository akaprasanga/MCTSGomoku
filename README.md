# 7343 Homework 2 AlphaZero:Gomoku

Your task in this homework is to implement the Monte Carlo Tree Search used in *AlphaZero* for the board game Gomoku (five in a line). For simplicity, we consider only the Gomoku game with a 11x11 board. 
1. You should implement a class named "MCTS" which must be a subclass of the class MCTSBase. The MCTSBase class already has the overall search process. In your MCTS class, you need to fill in the missing components by implementing (override) the abstract methods. (Do not change/override the non-abstract methods.) 
2. Your code also needs to implement another class (use any class name you want) that is a subclass of the TreeNode class and completes the implementation of the abstract methods in that class. 
3. The tree search process needs to utilize a deep neural network. Given a state s, the DNN  estimates the state value of s as well as computes the policy at s (the probability of actions). You should implement such a neural network and use it in the code for some of the abstract methods.     

 
 
