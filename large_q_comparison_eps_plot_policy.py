import numpy as np
import matplotlib.pyplot as plt
from maze_generator import WindyMazeEnv, read_maze
from mdp_graph import graph_value_policy

import matplotlib

############## Options to generate nice figures
fig_width_pt = 453.0 * 2  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean / 2.3  # height in inches
scale = 1.5
fig_size = [fig_width * scale, fig_height * scale]

############## Colors I like to use
my_yellow = [235. / 255, 164. / 255, 17. / 255]
my_blue = [58. / 255, 93. / 255, 163. / 255]
dark_gray = [68./255, 84. /255, 106./255]
my_red = [163. / 255, 93. / 255, 58. / 255]

my_color = dark_gray # pick color for theme

params_keynote = {
    'axes.labelsize': 16,
    'font.size': 16,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size
}
############## Parameters I use for IEEE papers
params_ieee = {
    'figure.autolayout' : True,
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    # 'text.usetex': True,
    # 'text.latex.preamble': '\\usepackage{sfmath}',
    'font.family': 'sans-serif',
    'figure.figsize': fig_size,
}

############## Choose parameters you like
matplotlib.rcParams.update(params_ieee)


maze_shape = (32, 32)
maze_file = 'maze/mazeLarge.png'

s = "20190414-185104-large-q-softmax-1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(141)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
plt.title("epsilon = 0.05")
# plt.colorbar()


s = "20190413-224746-large-q-epsilon-p1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)


plt.subplot(142)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
plt.title("epsilon = 0.1")

# plt.colorbar()

s = "20190413-224746-large-q-epsilon-p2.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)


# plt.show()

plt.subplot(143)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
plt.title("epsilon = 0.2")
# plt.savefig("../tex/figures/large_q_policy_comparison.pdf")

# plt.colorbar()
s = "20190414-191112-large-q-softmax-100.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(144)
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
plt.title("Boltzmann Exploration t=100")
plt.savefig("../tex/figures/large_q_policy_comparison.pdf")

# plt.colorbar()
plt.show()