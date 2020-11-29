#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload 2')
import numpy as np
import maze as mz
maze = np.array([[2, 0, 0, 0, 0, 2],
                 [0, 0, 0, 0, 0, 0], 
                 [2, 0, 0, 0, 0, 2]])
mz.draw_maze(maze)
env = mz.Maze(maze)


# In[5]:


gamma = 0.5;
epsilon = 1e-5;
# Solve the MDP problem with dynamic programming 
V, policy= mz.value_iteration(env, gamma, epsilon);
# Simulate the shortest path starting from position A
method = 'ValIter';
start  = (0, 0, 1, 2);
path = env.simulate(start, policy, method);


# In[3]:


print(path)

