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



maze_shape = (16, 16)
maze_file = 'maze/maze.png'


mynorm = plt.Normalize(vmin=0, vmax=10)
sm = plt.cm.ScalarMappable(cmap='plasma', norm=mynorm)

timestr = "20190412-150419-0p3"
s = "{}-small-q-a-p1.npy".format(timestr)
s = "20190414-195601-0p1-small-q-epsilon-p1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(141)
plt.title("p = 0.1 epsilon-greedy")
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
# plt.colorbar()

s = "20190414-192202-small-q-softmax-100.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(142)
plt.title("p = 0.1 Boltzmann exploration")
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
# plt.colorbar()


timestr = "20190413-014457-0p3"
s = "{}-small-q-epsilon-p1.npy".format(timestr)
s = "20190414-195601-0p3-small-q-epsilon-p1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)


plt.subplot(143)
plt.title("p = 0.3")
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)

# plt.colorbar()
# s = "{}-small-q-a-p2.npy".format(timestr)
s = "20190414-195601-0p5-small-q-epsilon-p1.npy"
q = np.load("data/{}".format(s))
policy = np.argmax(q, axis=1)

plt.subplot(144)
plt.title("p = 0.5")
value = np.max(q, axis=1)
policy = policy.reshape(maze_shape)
value = value.reshape(maze_shape)
maze = read_maze(maze_file)
im = graph_value_policy(value, policy, maze)
# plt.colorbar()

plt.savefig("../tex/figures/q_policy_comparison.pdf")



plt.show()