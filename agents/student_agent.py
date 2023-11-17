# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time

class Node(object):
    def __init__(self, parent, action):
        #self.name=key
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
        for (move, dir) in allowed_moves:
            child = Node(root, (move, dir))
            root.addChild(child)
        return 

    def set_barrier(self, pos, dir, chess_board):
        r, c = pos
        chess_board[r, c, dir] = True
        move = self.moves[dir]
        chess_board[r + move[0], c + move[1], self.opposites[dir]] = True
        
    def check_boundary(self, pos):
        r, c = pos
        return 0 <= r < self.chess_board.shape[0] and 0 <= c < self.chess_board.shape[0]
    
    def get_array_first_move(self):
        # Get position of the adversary 
        adv_pos = self.adv_pos

        # BFS
        state_queue = [(self.my_pos, 0)]
        allowed_moves = []
        
        for dir, move in enumerate(self.moves):
            r, c = self.my_pos
            if self.chess_board[r, c, dir]:
                continue
            allowed_moves.append((tuple(self.my_pos), dir))
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                continue
            for dir, move in enumerate(self.moves):
                next_pos = tuple(map(lambda x, y: x + y, cur_pos, move))
                if not self.check_boundary(next_pos): #check if the move moves outside the boundary
                    continue
                if self.chess_board[r, c, dir]: #check if there is a wall in the way
                    continue
                if np.array_equal(next_pos, adv_pos): #check if the adversary is in the way
                    continue
                if (tuple(next_pos), dir) in allowed_moves: #check if we already have the move
                    continue
                allowed_moves.append((tuple(next_pos), dir))
                state_queue.append((next_pos, cur_step + 1))

        return allowed_moves
        
    def run_simulation(self, chess_board, my_pos, adv_pos, max_step):
        #chess_board = c_board.copy()
        #moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        #opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        p1 = RandomAgent()
        #ended, s1, s2 = self.check_endgame(chess_board, my_pos, adv_pos)
        pos_p1 = adv_pos    #p1 is the enemy making a move first 
        pos_p2 = my_pos   #p2 is us making out move second
        while True:
            #take a step, update the chess board and then check if the game ended
            pos_p1, dir_p1 = p1.step(chess_board, pos_p1, pos_p2, max_step)
            '''
            try:
                pos_p1, dir_p1 = p1.step(chess_board, pos_p1, pos_p2, max_step)
            except:
                return 0
            '''
            #r, c = pos_p1
            #chess_board[r, c, dir_p1] = True
            #move = moves[dir_p1]
            #chess_board[r + move[0], c + move[1], opposites[dir_p1]] = True
            self.set_barrier(pos_p1, dir_p1, chess_board)
            #print('chess_board agter update:',chess_board)
            ended, s1, s2 = self.check_endgame(chess_board, pos_p1, pos_p2)
            if ended:
                if s1 > s2: return 0
                elif s1 == s2: return 0.5
                else: return 1
            pos_p2, dir_p2 = p1.step(chess_board, pos_p2, pos_p1, max_step)
            '''
            try:
                pos, dir = p1.step(chess_board, pos_p2, pos_p1, max_step)
            except:
                return 0
                '''
            #print('Before change:', chess_board[pos_p2[0]][pos_p2[1]][dir_p2])
            #r, c = pos_p2
            #chess_board[r, c, dir_p2] = True
            #move = moves[dir_p2]
            #chess_board[r + move[0], c + move[1], opposites[dir_p2]] = True
            #print('after change', chess_board[pos_p2[0]][pos_p2[1]][dir_p2])
            self.set_barrier(pos_p2, dir_p2, chess_board)

            ended, s1, s2 = self.check_endgame(chess_board, pos_p1, pos_p2)
            if ended:
                if s1 > s2: return 0
                elif s1 == s2: return 0.5
                else: return 1
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
        #print(chess_board)
        #print(chess_board.shape())
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
        for n in node.children:
            print(n)
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

        for n in root.children:
            c_board = self.chess_board.copy()
            self.set_barrier(n.move, n.wall_place, c_board)
            res = self.run_simulation(c_board, n.move, self.adv_pos, self.max_step)
            self.back_prop(n, res)

        start_time = time.time()
        time_taken = time.time() - start_time
        while time_taken < 1.95:
            n = self.choose_next_expansion(root)
            c_board = self.chess_board.copy()
            self.set_barrier(n.move, n.wall_place, c_board)
            res = self.run_simulation(c_board, n.move, self.adv_pos, self.max_step)
            self.back_prop(n, res)
            time_taken = time.time() - start_time
        
        best_choice = None
        best_value = 0
        for n in root.children:
            if n.value/n.visits > best_value:
                best_value = n.value/n.visits
                best_choice = n

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
        start_time = time.time()
        time_taken = time.time() - start_time
        sims = 0
        root = Node(None, (my_pos, None))
        uct_tree = MCTS(root, chess_board, my_pos, adv_pos, max_step)
        uct_tree.set_children(root)
        output = uct_tree.find_best_move()

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
