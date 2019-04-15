import gym
from maze_generator import MazeEnv, read_maze
from value_iteration import value_iteration
from mdp_graph import graph_value_policy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib






############## Options to generate nice figures
fig_width_pt = 453.0 * 2  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean / 2.3  # height in inches
scale = 2
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

# env = gym.make("FrozenLake-v0")
maze_file = 'maze/mazeLarge.png'

def val_run(discount_factor=0.9):
    maze = read_maze(maze_file)
    env = MazeEnv(maze_file=maze_file)
    obs = env.reset()
    policy, V, num_iter, total_access = value_iteration(env, discount_factor=discount_factor)

    coord, action = np.where(policy == 1)
    policy = action.reshape(maze_shape)
    V = V.reshape(maze_shape)
    im = graph_value_policy(V, policy, maze)
    return im

fig, _ = plt.subplots(1, 4)
ax = plt.subplot(141)
ax.set_title("0.90")
val_run(0.9)

ax = plt.subplot(142)
ax.set_title("0.95")
val_run(0.95)

ax = plt.subplot(143)
ax.set_title("0.97")
val_run(0.97)

ax = plt.subplot(144)
ax.set_title("0.99")
im = val_run(0.99)

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)

# plt.savefig("../tex/figures/value_iterations_small_p.pdf")
plt.show()

