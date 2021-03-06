import numpy as np
import matplotlib.pyplot as plt
import matplotlib

############## Options to generate nice figures
fig_width_pt = 453.0 * 2  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean / 2  # height in inches
fig_size = [fig_width, fig_height]

############## Colors I like to use
my_yellow = [235. / 255, 164. / 255, 17. / 255]
my_blue = [58. / 255, 93. / 255, 163. / 255]
dark_gray = [68./255, 84. /255, 106./255]
my_red = [163. / 255, 93. / 255, 58. / 255]

my_color = dark_gray # pick color for theme
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



plt.subplot(121)
timestr = "20190413-224855"
timestr1 = "20190413-224746"
timestr2 = "20190413-224608"

s = "{}-large-q_start_hist-epsilon-p1.npy".format(timestr)
q1 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p05.npy".format(timestr)
q2 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p2.npy".format(timestr)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, label='epsilon=0.05', c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, label='epsilon=0.1', c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, label='epsilon=0.2', c='C2', alpha=0.9)

s = "20190414-191112-large-q_start_hist-softmax-100.npy"
q3 = np.load("data/{}".format(s))
plt.plot(range(len(q3)), q3, label='Boltzmann Exploration', c='C3', alpha=0.9)


s = "{}-large-q_start_hist-epsilon-p1.npy".format(timestr1)
q1 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p05.npy".format(timestr1)
q2 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p2.npy".format(timestr1)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, c='C2', alpha=0.9)

s = "{}-large-q_start_hist-epsilon-p1.npy".format(timestr2)
q1 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p05.npy".format(timestr2)
q2 = np.load("data/{}".format(s))

s = "{}-large-q_start_hist-epsilon-p2.npy".format(timestr2)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, c='C2', alpha=0.9)


plt.legend()
plt.ylabel("Max Q Value of Start State")
plt.xlabel("Episodes")

plt.subplot(122)
s = "{}-large-total_reward_hist-epsilon-p1.npy".format(timestr)
q1 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p05.npy".format(timestr)
q2 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p2.npy".format(timestr)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, c='C2', alpha=0.9)

s = "{}-large-total_reward_hist-epsilon-p1.npy".format(timestr1)
q1 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p05.npy".format(timestr1)
q2 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p2.npy".format(timestr1)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, c='C2', alpha=0.9)

s = "{}-large-total_reward_hist-epsilon-p1.npy".format(timestr2)
q1 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p05.npy".format(timestr2)
q2 = np.load("data/{}".format(s))

s = "{}-large-total_reward_hist-epsilon-p2.npy".format(timestr2)
q3 = np.load("data/{}".format(s))

plt.plot(range(len(q1)), q2, label='epsilon=0.05', c='C0', alpha=0.9)
plt.plot(range(len(q1)), q1, label='epsilon=0.1', c='C1', alpha=0.9)
plt.plot(range(len(q1)), q3, label='epsilon=0.2', c='C2', alpha=0.9)

s = "20190414-191112-large-total_reward_hist-softmax-100.npy"
q3 = np.load("data/{}".format(s))
plt.plot(range(len(q3)), q3, label='Boltzmann Exploration', c='C3', alpha=0.9)



plt.legend()

plt.ylabel("Cumulative Discounted Total Reward")
plt.xlabel("Episodes")

plt.savefig("../tex/figures/large_q_eps_hist.pdf")


plt.show()

# s = "20190413-220048-large-q_start_hist-epsilon-p1.npy"
# q1 = np.load("data/{}".format(s))

# s = "20190413-220048-large-q_start_hist-epsilon-p05.npy"
# q2 = np.load("data/{}".format(s))

# s = "20190413-220048-large-q_start_hist-epsilon-p2.npy"
# q3 = np.load("data/{}".format(s))

# plt.plot(range(len(q1)), q2, label='epsilon=0.05', c='C0', alpha=0.9)
# plt.plot(range(len(q1)), q1, label='epsilon=0.1', c='C1', alpha=0.9)
# plt.plot(range(len(q1)), q3, label='epsilon=0.2', c='C2', alpha=0.9)

# plt.legend()
# plt.show()