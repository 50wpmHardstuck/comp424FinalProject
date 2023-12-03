# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

class Node(object):
    def __init__(self, parent, action):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = action[0]          
        self.wall_place = action[1]
        
    def addChild(self, node):
        self.children.append(node)
    

class MCTS(object):
    def __init__(self, root, chess_board, my_pos, adv_pos, max_steps):
        self.root = root
        self.chess_board = chess_board
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        self.max_step = max_steps
        
    def set_children(self, root):
        allowed_moves = self.get_array_first_move()
        #print(len(allowed_moves))
        #if len(allowed_moves) == 1:
           # print(allowed_moves)
          #  child = Node(root, (allowed_moves[0][0], allowed_moves[0][1]))
          #  root.addChild(child)
          #  print(len(root.children))
          #  return
        for (move, dir) in allowed_moves:
            child = Node(root, (move, dir))
            root.addChild(child)
        print(len(root.children))
        return 

    def set_barrier(self, pos, dir, chess_board):
        r, c = pos
        chess_board[r, c, dir] = True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def unset_barrier(self, pos, dir, chess_board):
        r, c = pos
        chess_board[r, c, dir] = False
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = False
        
    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.chess_board.shape[0] and 0 <= c < self.chess_board.shape[0]
    
    def get_array_first_move(self):
        adv_pos = self.adv_pos
        
        state_queue = [(self.my_pos, 0)]
        allowed_moves = []
        
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            for dir, move in enumerate(self.moves):
                next_pos = tuple(map(lambda x, y: x + y, cur_pos, move))
                r_new, c_new = next_pos
                if self.chess_board[r, c, dir]:
                    continue
                if (tuple(cur_pos), dir) in allowed_moves: #check if we already have the move
                    break
                allowed_moves.append((tuple(cur_pos), dir))
                if cur_step == self.max_step:
                    continue
                if not self.check_boundary(next_pos): #check if the move moves outside the boundary
                    continue
                if np.array_equal(next_pos, adv_pos): #check if the adversary is in the way
                    continue
                state_queue.append((tuple(next_pos), cur_step + 1))
                
        return allowed_moves
        
    def run_simulation(self, chess_board, my_pos, adv_pos, max_step):
        #print('starting a sim')
        p1 = RandomAgent()
        pos_p1 = adv_pos    #p1 is the enemy making a move first 
        pos_p2 = my_pos   #p2 is us making out move second
        ended, s1, s2 = self.check_endgame(chess_board, pos_p1, pos_p2)
        if ended:
                if s1 > s2: return 0
                elif s1 == s2: return 0.5
                else: return 1

        while True:
            #take a step, update the chess board and then check if the game ended
            for i in range(0, 5):
                #print(i)
                prev_p1_pos = pos_p1
                #print('adv move')
                pos_p1, dir_p1 = p1.step(chess_board, pos_p1, pos_p2, max_step)
                self.set_barrier(pos_p1, dir_p1, chess_board)
                ended, s1, s2 = self.check_endgame(chess_board, pos_p1, pos_p2)
                if ended:
                    #print('in end step')
                    #print(s1, s2)
                    #print(i)
                    #print('after adv move, adv score:', s1, 'my score:', s2)
                    if i == 4:
                        return s1 < s2
                    if s1 < 3 and i < 4: 
                        #print('tried step again')
                        self.unset_barrier(pos_p1, dir_p1, chess_board)
                        pos_p1 = prev_p1_pos     
                        continue
                    elif s1 > s2: 
                        #print('lost') 
                        return 0
                    elif s1 == s2: 
                        #print('draw')
                        return 0.5
                    else: 
                        #print('win')
                        return 1
                break
            #print('actually switched to other loop')
            for j in range(0, 5):
                #print(i)
                prev_pos_p2 = pos_p2
                #print('my move')
                pos_p2, dir_p2 = p1.step(chess_board, pos_p2, pos_p1, max_step)
                self.set_barrier(pos_p2, dir_p2, chess_board)
                ended, s1, s2 = self.check_endgame(chess_board, pos_p1, pos_p2)
                if ended:
                    #print('in end step')
                    #print(s1, s2)
                    #print(j)
                    #print('after my move, adv score:', s1, 'my score:', s2)
                    if j == 4:
                        return s1 < s2
                    if s2 < 3 and j < 4:
                        #print('tried step again')
                        self.unset_barrier(pos_p2, dir_p2, chess_board)
                        pos_p2 = prev_pos_p2
                        continue
                    elif s1 > s2: 
                        #print('lost') 
                        return 0
                    elif s1 == s2: 
                        #print('draw')
                        return 0.5
                    else: 
                        #print('win')
                        return 1
                break
        print("This should not be happening!!!, ERROR")
        return -1 #should never occur, error
    
    def check_endgame(self, chess_board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        #Variables that we need to implement the copied function (added more in the signature of the function) (used to be self)
        board_size = chess_board.shape
        board_size = board_size[0]
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(moves[1:3]):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        
        return True, p0_score, p1_score
    
    def get_value(self, node):
        c = 1.41421 #sqrt of 2
        e = 2.71828 #e
        def log_approx(x):
            return 100*(x**(1/100))-100
        exploit = node.value/node.visits
        #have to do a weird approximation of log with exponent of 1/e
        explore = c * (log_approx(node.parent.visits)/(node.visits))**(1/2) #c*sqrt(log_e(parent visits)/node visits)
        return exploit + explore


    def choose_next_expansion(self, node):
        best_score = 0
        best_n = "not getting reset"
        if len(node.children) == 1:
            return node.children[0]
        for n in node.children:
            #print(n)
            if self.get_value(n) > best_score:
                best_score = self.get_value(n)
                best_n = n
        return best_n
    
    def back_prop(self, node, value):
        if node != self.root:
            node.value += value
            node.visits += 1
            return self.back_prop(node.parent, value)
        else:
            node.value += value
            node.visits += 1
            return

    def find_best_move(self):
        root = self.root
        start_time = time.time()
        time_taken = time.time() - start_time               
        for ind, n in enumerate(root.children):
            #print(ind)
            c_board = self.chess_board.copy()
            self.set_barrier(n.move, n.wall_place, c_board)
            res = self.run_simulation(c_board, n.move, self.adv_pos, self.max_step)
            self.back_prop(n, res)

        #time issue on 12x12 is that we dont have enough time to visit eaach child once, maybe we should use heuristics
        
        while time_taken <= 1.95:
            n = self.choose_next_expansion(root)
            c_board = self.chess_board.copy()
            self.set_barrier(n.move, n.wall_place, c_board)
            res = self.run_simulation(c_board, n.move, self.adv_pos, self.max_step)
            self.back_prop(n, res)
            
            time_taken = time.time() - start_time
        counter = 0
        best_winrate = 0
        best_choice = None
        if len(root.children) == 1:
            best_choice = root.children[0]
            return best_choice.move, best_choice.wall_place
        for i in root.children:
            #print('node counter:', counter, i.value/i.visits, i.visits)
            counter += 1
            if i.value/i.visits > best_winrate:
                best_choice = i
                best_winrate = i.value/i.visits
        print('best winrate:', best_choice.value/best_choice.visits)
        print('best node visits:', best_choice.visits)
        return best_choice.move, best_choice.wall_place


    
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
        try:
            assert len(allowed_barriers)>=1 
        except:
            print('my_pos', my_pos)
            print('adv_pos', adv_pos)
            print(chess_board)
            assert len(allowed_barriers)>=1 
        dir = allowed_barriers[np.random.randint(0, len(allowed_barriers))]
        #print(allowed_dirs)
        #print(my_pos, dir)
        return my_pos, dir


@register_agent("improved_agent")
class BetterStudent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(BetterStudent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
    
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
        #p1 = RandomAgent()
        #output = p1.step(chess_board, my_pos, adv_pos, max_step)
        #print(chess_board)
        start_time = time.time()
        #time_taken = time.time() - start_time
        sims = 0
        root = Node(None, (my_pos, None))
        uct_tree = MCTS(root, chess_board, my_pos, adv_pos, max_step)
        uct_tree.set_children(root)
        time_taken = time.time() - start_time
        print(time_taken,'setup time')
        output = uct_tree.find_best_move()
        print(time.time()-start_time, 'time to find best move')                         

        '''while time_taken < 1.95:
            chess_board_copy = np.copy(chess_board)
            self.run_simulation(chess_board_copy, my_pos, adv_pos, max_step)
            sims += 1
            #print(sims)
            time_taken = time.time()-start_time'''
        
        print('NUMBER OF SIMS:', uct_tree.root.visits)
        time_taken = time.time() - start_time

        print("My AI's turn took ", time_taken, "seconds.")

        # dummy return
        return output #my_pos, self.dir_map["u"]