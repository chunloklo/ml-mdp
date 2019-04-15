import matplotlib.pyplot as plt
import numpy as np
import matplotlib


############## Options to generate nice figures
fig_width_pt = 453.0 * 2  # Get this from LaTeX using \showthe\column-width
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width * golden_mean / 2.5  # height in inches
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
plt.subplots(13)

discount_factors = np.arange(0.8, 1.0, 0.02)
value_iter = [23, 25, 27, 31, 36, 43, 54, 70, 100, 181]
policy_iter = [18, 20, 20, 20, 20, 19, 19, 19, 18, 17]
# value_iter = [3840, 3840, 4096, 4608, 5120, 5632, 6656, 8448, 12288, 22528]
policy_update = [430, 525, 575, 637, 720, 779, 917, 1089, 1287, 1698]


plt.subplot(131)
plt.title("# of Iterations Till Convergence")
plt.plot(discount_factors, value_iter, marker=".", label='Value Iteration')
plt.plot(discount_factors, policy_iter, marker=".", label='Policy Iteration')
# plt.plot(discount_factors, policy_update, marker=".", label='Policy Iteration, Value Function Updates')
plt.xticks(discount_factors)
plt.xlabel("Discount Factor")
plt.ylabel("Iterations")
plt.ylim(0, 100)
plt.legend()


value_time = [0.07975649833679199, 0.08674979209899902, 0.09275126457214355, 0.10671520233154297, 0.12466621398925781, 0.14760541915893555, 0.18350839614868164, 0.23936152458190918, 0.3390927314758301, 0.6103684902191162]
policy_time = [1.0417239665985107, 1.2656147480010986, 1.3813071250915527, 1.533898115158081, 1.6914751529693604, 1.8181378841400146, 2.1313018798828125, 2.5262444019317627, 2.9660866260528564, 3.874619722366333]
discount_factors = np.arange(0.8, 1.0, 0.02)
plt.subplot(132)
plt.title("Seconds Till Convergence")
plt.plot(discount_factors, value_time, marker=".", label='Value Iteration')
plt.plot(discount_factors, policy_time, marker=".", label='Policy Iteration')
plt.yscale('log')
plt.xticks(discount_factors)
plt.xlabel("Discount Factor")
plt.ylabel("Seconds")
plt.legend()


Ps = np.arange(0.1, 1.0, 0.1)
value_iter = [26, 34, 43, 50, 43, 35, 30, 21, 13]
policy_iter = [15, 14, 14, 16, 19, 9, 7, 5, 4]
plt.subplot(133)
plt.title("# of Iterations Till Convergence")
plt.plot(Ps, value_iter, marker=".", label='Value Iteration')
plt.plot(Ps, policy_iter, marker=".", label='Policy Iteration')
plt.xticks(Ps)
plt.xlabel("Wind Probability")
plt.ylabel("Iterations")
plt.legend()






plt.savefig("../tex/figures/small_p_comparison.pdf")

plt.show()