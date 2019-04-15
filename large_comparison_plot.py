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
value_iter = [48, 50, 52, 54, 58, 68, 79, 94, 115, 147]
policy_iter = [29, 29, 32, 29, 29, 29, 29, 29, 30, 32]
value_update = [48, 50, 52, 54, 58, 68, 79, 94, 115, 147]
policy_update = [1238, 1313, 1538, 1473, 1610, 1790, 2051, 2416, 3103, 4453]


plt.subplot(131)
plt.title("# of Iterations Till Convergence")
plt.plot(discount_factors, value_iter, marker=".", label='Value Iteration')
plt.plot(discount_factors, policy_iter, marker=".", label='Policy Iteration')
# plt.plot(discount_factors, policy_update, marker=".", label='Policy Iteration, Value Function Updates')
plt.xticks(discount_factors)
plt.xlabel("Discount Factor")
plt.ylabel("Iterations")
# plt.ylim(0, 100)
plt.legend()


value_time = [0.567502498626709, 0.5893881320953369, 0.6143465042114258, 0.6363260746002197, 0.6861634254455566, 0.7888908386230469, 0.9285149574279785, 1.10603928565979, 1.3763182163238525, 1.725414514541626]
policy_time = [9.304095029830933, 9.928442478179932, 11.64685869216919, 11.048440217971802, 12.046789169311523, 13.347302436828613, 15.341986894607544, 18.0228111743927, 23.0144624710083, 32.913006067276]
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


Ps = np.arange(0.8, 1.0, 0.02)
value_iter = [26, 34, 43, 50, 43, 35, 30, 21, 13]
policy_iter = [15, 14, 14, 16, 19, 9, 7, 5, 4]
plt.subplot(133)
plt.title("# of State Updates Till Convergence")
plt.plot(Ps, value_update, marker=".", label='Value Iteration')
plt.plot(Ps, policy_update, marker=".", label='Policy Iteration')
plt.xticks(Ps)
plt.yscale('log')
plt.xlabel("Discount Factor")
plt.ylabel("# of State Updates")
plt.legend()






plt.savefig("../tex/figures/large_p_comparison.pdf")

plt.show()