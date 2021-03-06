import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 10
    CAUGHT_REWARD = -50
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, start=(0,0,1,2), weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.police_actions = []
        self.start = start;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);
        

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for im in range(self.maze.shape[0]):
                    for jm in range (self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = (i, j, im, jm);
                            map[(i, j, im, jm)] = s;
                            s += 1;
        return states, map
    
    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """        
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1])
        # Based on the impossiblity check return the next state.
        
        if hitting_maze_walls:
            row, col = self.states[state][:2]
        n_state = self.map[(row, col, self.states[state][2] , self.states[state][3])]
        
        return n_state

    def __move_minotaur(self, state, maction=None):
        """ Makes a step in the maze given a current state and action
            available actions depend on wether he can stay or not
            """
        ar, ac , pr, pc = self.states[state]
        
        if(maction is None):
            
            if ar == pr:
                if ac > pc:
                    self.police_actions = [2, 3, 4]
                elif ac< pc:
                    self.police_actions = [1, 3, 4]
            elif ac == pc:
                if ar > pr:
                    self.police_actions = [1, 2, 3]
                elif pr > ar:
                    self.police_actions = [1, 2, 4]
            elif pr > ar:
                if pc > ac:
                    self.police_actions = [1, 3]
                elif ac > pc:
                    self.police_actions = [2, 3]   
            elif pr < ar:
                if pc > ac:
                    self.police_actions =    [1, 4] 
                elif ac > pc:
                    self.police_actions =    [2, 4] 
            
        possible_n_states = []
        for p_action in self.police_actions:
            p_row = self.states[state][2] +  self.actions[p_action][0]
            p_col = self.states[state][3] +  self.actions[p_action][1]
            
            hitting_maze_walls =  (p_row == -1) or (p_row == self.maze.shape[0]) or \
                                  (p_col == -1) or (p_col == self.maze.shape[1]) 
                    
            if self.states[state][:2] == self.states[state][2:4]:
                print("Confirm caught")
                possible_n_states.append(self.map[(0,0,1,2)])
                continue
            
            if hitting_maze_walls:
                p_row = self.states[state][2]
                p_col = self.states[state][3]
            possible_n_states.append(self.map[(ar, ac, p_row, p_col)])
            
        return possible_n_states
          
    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                if self.states[s][:2] == self.states[s][2:4]:
                        transition_probabilities[self.map[self.start], s, a] = 1;
                else:       
                    next_s = self.__move(s,a);
                    police_move_states = self.__move_minotaur(next_s);
                    for potential_s in police_move_states:
                        transition_probabilities[potential_s, s, a] = 1/len(police_move_states);
                    
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    a_pos = self.states[s][:2]
                    next_s = self.__move(s,a);
                    next_p_states = self.__move_minotaur(next_s)
                    
                    c_reward = 0
                    prob = 1 / len(next_p_states)
                    
                    for p_state in next_p_states:
                        next_a_pos = self.states[p_state][0:2];
                        next_p_pos = self.states[p_state][2:4];
  
                        # Rewrd for hitting a wall
                        if next_a_pos == a_pos and a != self.STAY:
                            c_reward +=  self.IMPOSSIBLE_REWARD;
                        elif  2 == self.maze[next_a_pos]:
                            c_reward +=  self.GOAL_REWARD;
                        elif next_a_pos == next_p_pos:
                            c_reward +=  self.CAUGHT_REWARD;
                    
                    rewards[s,a] = prob * c_reward


        return rewards;

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                police_possible_states = self.__move_minotaur(next_s)
                next_s = police_possible_states[np.random.randint(0, len(police_possible_states))]
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            police_possible_states = self.__move_minotaur(next_s)
            next_s = police_possible_states[np.random.randint(0, len(police_possible_states))]                                               
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while True:
                print("start", self.states[s], self.states[next_s])
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                                                                     
                police_possible_states = self.__move_minotaur(next_s)
                print("Possible states:" ,police_possible_states)
                next_s = police_possible_states[np.random.randint(0, len(police_possible_states))]                                          
                                                                     
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                print("end", self.states[s], self.states[next_s])
        return path


    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));


    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1);
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1);
    return V, policy;

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

        
    # Update the color at each frame
    for i in range(len(path)):
        print(path[i])
        grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0:2])].get_text().set_text('Robber')
        grid.get_celld()[(path[i][2:4])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:4])].get_text().set_text('Police')
        if i > 0:
            print(i)
            text_a = "a: "
            text_m = "m: "
            if path[i][:2] == path[i-1][:2]:
                text_a += str(i)
                grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[maze[path[i-1][0:2]]])
                grid.get_celld()[(path[i-1][0:2])].get_text().set_text(text_a)
            if path[i][2:4] == path[i-1][2:4]:
                text_m += str(i)
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text(text_m)
            else:
                grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[maze[path[i-1][0:2]]])
                grid.get_celld()[(path[i-1][0:2])].get_text().set_text('agent: ' + str(i))
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('minot.: '+ str(i))
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
