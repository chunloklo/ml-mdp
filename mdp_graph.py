import matplotlib.pyplot as plt

def show_values(cm, maze, fmt="%.2f", **kw):
    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            dx = 0
            dy = 0
            length = 1
            if cm[i][j] == 0:
                dx = -length
            if cm[i][j] == 1:
                dy = length
            if cm[i][j] == 2:
                dx = length
            if cm[i][j] == 3:
                dy = -length

            # plt.arrow(j, i, dx, dy
            if (maze[i, j] not in b'BGTM'):
                plt.annotate("", xy=(j + dx / 2, i + dy / 2), xytext=(j - dx / 2, i - dy / 2), arrowprops=dict(color='white', arrowstyle="->"))
            # plt.text(j, i, format(cm[i, j], fmt),
            #         ha="center", va="center",
            #         color="white" if cm[i, j] > thresh else "black")

def graph_value_policy(value, policy, maze):
    plt.axis('off')

    # im = plt.imshow(value, cmap='viridis')
    im = plt.imshow(value, cmap='viridis', vmin = -1, vmax=1)
    # plt.colorbar()
    show_values(policy, maze)
    return im