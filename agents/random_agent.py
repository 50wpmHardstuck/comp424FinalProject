import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent

# Important: you should register your agent with a name
@register_agent("random_agent")
class RandomAgent(Agent):
    """
    Example of an agent which takes random decisions
    """

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.autoplay = True
        
    def check_boundary(self, pos, chess_board):
        r, c = pos
        return 0 <= r < chess_board.shape[0] and 0 <= c < chess_board.shape[0]
        
    def get_array_first_move(self, my_pos, adv_pos, chess_board, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        state_queue = [(my_pos, 0)]
        allowed_moves = []

        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            for dir, move in enumerate(moves):
                next_pos = tuple(map(lambda x, y: x + y, cur_pos, move))
                if chess_board[r, c, dir]:
                    continue
                if (tuple(cur_pos), dir) in allowed_moves: #check if we already have the move
                    break
                allowed_moves.append((tuple(cur_pos), dir))
                if cur_step == max_step:
                    continue
                if not self.check_boundary(next_pos, chess_board): #check if the move moves outside the boundary
                    continue
                if np.array_equal(next_pos, adv_pos): #check if the adversary is in the way
                    continue
                state_queue.append((tuple(next_pos), cur_step + 1))

        return allowed_moves
    
    def check_one_move_win(self, chess_board, my_pos, adv_pos, max_step):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        reverse_wall = {0:2, 1:3, 2:0, 3:1}
        r, c = adv_pos
        dir0 = int(chess_board[r, c, 0])
        dir1 = int(chess_board[r, c, 1])
        dir2 = int(chess_board[r, c, 2])
        dir3 = int(chess_board[r, c, 3])
        wall_to_build = 0
        if(dir0+dir1+dir2+dir3 == 3):
            if dir1 == False:
                wall_to_build = 1
            elif dir2 == False:
                wall_to_build = 2
            elif dir3 == False:
                wall_to_build = 3
            move = moves[wall_to_build]
            pos_to_block = tuple(map(lambda x, y: x + y, adv_pos, move))
            possible_moves = self.get_array_first_move(my_pos, adv_pos, chess_board, max_step)
            if (pos_to_block, wall_to_build) in possible_moves:
                return pos_to_block, reverse_wall[wall_to_build]
            else:
                return 0
        return 0
    
    def step(self, chess_board, my_pos, adv_pos, max_step):
        can_block = self.check_one_move_win(chess_board, my_pos, adv_pos, max_step)
        if can_block != 0:
            my_pos, dir = can_block
            return my_pos, dir
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        r, c = my_pos
        count_walls_start = int(chess_board[r, c, 0]) + int(chess_board[r, c, 1]) + int(chess_board[r, c, 2]) + int(chess_board[r, c, 3])
        if (count_walls_start < 3):
            steps = np.random.randint(0, max_step + 1)
        else:
            steps = np.random.randint(1, max_step + 1)
        #steps = np.random.randint(0, max_step + 1)
        # Pick steps random but allowable moves
        count_walls = 4
        break_loop = False
        r_new, c_new = my_pos
        og_pos = my_pos
        #allowed_tries = 0
        while(True):
            #allowed_tries += 1
            my_pos = og_pos
            for _ in range(steps):
                r, c = my_pos
                # Build a list of the moves we can make
                allowed_dirs = [ d                                
                    for d in range(0,4)                           # 4 moves possible
                    if not chess_board[r,c,d] and                 # chess_board True means wall
                    not adv_pos == (r+moves[d][0],c+moves[d][1])] # cannot move through Adversary
                if len(allowed_dirs)==0:
                    # If no possible move, we must be enclosed by our Adversary
                    break_loop = True
                    break
                random_dir = allowed_dirs[np.random.randint(0, len(allowed_dirs))]
                # This is how to update a row,col by the entries in moves 
                # to be consistent with game logic
                m_r, m_c = moves[random_dir]
                my_pos = (r + m_r, c + m_c)
                r_new = r+m_r
                c_new = c+m_c  
                #print(int(chess_board[r_new, c_new, 0]))
            
            if(break_loop == True):
                break
            #print(chess_board[r_new, c_new, 0], chess_board[r_new, c_new, 1], chess_board[r_new, c_new, 2], chess_board[r_new, c_new, 3])
            count_walls = int(chess_board[r_new, c_new, 0]) + int(chess_board[r_new, c_new, 1]) + int(chess_board[r_new, c_new, 2]) + int(chess_board[r_new, c_new, 3])
            if (count_walls < 3):
                break
            if (count_walls_start < 3):
                steps = np.random.randint(0, max_step + 1)
            else:
                steps = np.random.randint(1, max_step + 1)

        # Final portion, pick where to put our new barrier, at random
        r, c = my_pos
        # Possibilities, any direction such that chess_board is False
        allowed_barriers=[i for i in range(0,4) if not chess_board[r,c,i]]
        # Sanity check, no way to be fully enclosed in a square, else game already ended
        assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]

        return my_pos, dir

    def step_ayo(self, chess_board, my_pos, adv_pos, max_step):
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
        #print(allowed_dirs)
        return my_pos, dir
