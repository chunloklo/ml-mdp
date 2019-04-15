import numpy as np
import matplotlib.pyplot as plt
from maze_generator import WindyMazeEnv, read_maze
from mdp_graph import graph_value_policy

maze_shape = (16, 16)
maze_file = 'maze/maze.png'

s = "20190412-141228-small-q-epsilon-p1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(141)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

s = "20190412-141248-small-q-epsilon-p05.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(142)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

s = "20190412-141306-small-q-epsilon-p2.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(143)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

s = "20190414-192202-small-q-softmax-100.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(144)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)


plt.show()