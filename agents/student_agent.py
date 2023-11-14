# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

class RandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        steps = np.random.randint(0, max_step + 1)

        # Pick steps random but allowable moves
        for _ in range(steps):
            r, c = my_pos

            # Build a list of the moves we can make
            allowed_dirs = [ d                                
                for d in range(0,4)                           # 4 moves possible
                if not chess_board[r,c,d] and                 # chess_board True means wall
                not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary

            if len(allowed_dirs)==0:
                # If no possible move, we must be enclosed by our Adversary
                break

            random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]

            # This is how to update a row,col by the entries in moves 
            # to be consistent with game logic
            m_r, m_c = moves[random_dir]
            my_pos = (r + m_r, c + m_c)

        # Final portion, pick where to put our new barrier, at random
        r, c = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

        return my_pos, dir


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
    def run_simulation(chess_board, my_pos, adv_pos, max_step):
        p1 = RandomAgent()
        ended, s1, s2 = check_endgame(len(chess_board), my_pos, adv_pos)
        counter = 0
        pos_p1 = my_pos     #p1 is the enemy making a move first 
        pos_p2 = adv_pos    #p2 is us making out move second
        while True:
            #take a step, update the chess board and then check if the game ended
            pos_p1, dir_p1 = p1.step(chess_board, pos_p1, pos_p2, max_step)
            chess_board[pos_p1[0]][pos_p1[1]][self.dir_map[dir_p1]] = True
            ended, s1, s2 = check_endgame(len(chess_board), pos_p1, pos_p2)
            if ended:
                if s1> s2: return 0
                else: return 1

            pos_p2, dir_p2 = p2.step(chess_board, pos_p2, pos_p1, max_step)
            chess_board[pos_p2[0]][pos_21[1]][self.dir_map[dir_p2]] = True
            ended, s1, s2 = check_endgame(len(chess_board), pos_p1, pos_p2)
            if ended:
                if s1 > s2: return 0
                else: return 1
        print("This should not be happening!!!, ERROR")
        return -1 #should never occur, error

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        
        p1 = RandomAgent()
        output = p1.step(chess_board, my_pos, adv_pos, max_step)
        
        start_time = time.time()
        time_taken = time.time() - start_time
        print('can i push')
        
        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return output #my_pos, self.dir_map["u"]
