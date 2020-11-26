#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import temp as mz
maze = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0], 
                 [0, 0, 1, 0, 0, 0, 1, 1, 0],
                 [0, 0, 1, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 1, 1, 1, 1, 1, 1, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 2, 0]])
#mz.draw_maze(maze)
env = mz.Maze(maze)
mz.draw_maze(maze)

# In[2]:


#print(env.transition_probabilities)
#print(env.show())


# In[3]:


# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming 
V, policy= mz.dynamic_programming(env,horizon);


# In[4]:


list1 = ["1", "2"]
list2 = (1,2)
dict1 = {list2:5}


# In[5]:


# Simulate the shortest path starting from position A
method = 'DynProg';
start  = (0,0, 6, 6);
path = env.simulate(start, policy, method);


# In[6]:


print(path)


# In[ ]:





# In[10]:


#mz.animate_solution(maze, path)
mz.draw_maze(maze)


# In[ ]:




