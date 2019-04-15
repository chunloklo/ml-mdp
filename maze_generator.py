from PIL import Image
import numpy as np


from gym.envs.toy_text import discrete
from gym import utils
import sys

def read_maze(file):
    im = Image.open(file) # Can be many different formats.
    pix = im.load()
    # print("Image Size:", im.size)  # Get the width and hight of the image for iterating over

    pixels = np.asarray(im)

    # print(im.size[0:2])
    desc = np.empty(im.size[0:2], dtype='c')

    # print(desc.shape)

    for r in range(im.size[0]):
        for c in range(im.size[1]):
            if np.array_equal(pixels[r,c],  [0, 0, 0]):
                desc[r, c] = 'B'
            elif np.array_equal(pixels[r,c],  [0, 0, 255]):
                desc[r, c] = 'S'
            elif np.array_equal(pixels[r,c],  [0, 255, 0]):
                desc[r, c] = 'G'
            elif np.array_equal(pixels[r,c],  [0, 255, 255]):
                desc[r, c] = 'T'
            elif np.array_equal(pixels[r,c],  [255, 0, 0]):
                desc[r, c] = 'R'
            elif np.array_equal(pixels[r,c],  [255, 255, 0]):
                desc[r, c] = 'Y'
            elif np.array_equal(pixels[r,c],  [255, 0, 255]):
                desc[r, c] = 'M'
            else:
                desc[r, c] = 'W'
    return desc

read_maze("maze/maze.png")

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class MazeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze_file):
        self.desc = desc = read_maze(maze_file)
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            prev_row = row
            prev_col = col
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            if self.desc[row, col] == b'B':
                row = prev_row
                col = prev_col
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)


                #start state
                letter = desc[row, col]
                if letter == b'S':
                    self.start = s

                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter == b'B':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1.0, to_s(row, col), -1.0, True))
                        pass
                    if letter == b'Y':
                        # newrow, newcol = inc(row, col, a)
                        # newstate = to_s(newrow, newcol)
                        # li.append((1, newstate, 0.0, False))

                        # newrow, newcol = inc(row, col, a)
                        # newstate = to_s(newrow, newcol)
                        # newletter = desc[newrow, newcol]
                        # li.append((1 - 0.5, newstate, 0, False))

                        # #going up
                        # newrow, newcol = inc(row, col, 3)
                        # newstate = to_s(newrow, newcol)
                        # newletter = desc[newrow, newcol]
                        # li.append((0.5, newstate, 0, False))

                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            li.append((1.0/3.0, newstate, 0, False))

                        # for b in range(4):
                        #     newrow, newcol = inc(row, col, b)
                        #     newstate = to_s(newrow, newcol)
                        #     newletter = desc[newrow, newcol]
                        #     li.append((1.0/4.0, newstate, 0, False))

                    if letter == b'M':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, to_s(row, col), 0.75, True))

                    if letter == b'T':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, to_s(row, col), 1.0, True))

                    if letter == b'G':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, to_s(row, col), 0.1, True))
                    # if letter == b'T':
                    #     newrow, newcol = inc(row, col, a)
                    #     newstate = to_s(newrow, newcol)
                    #     newletter = desc[newrow, newcol]
                    #     li.append((1, newstate, 1.0, False))
                    if letter == b'R':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        li.append((1, newstate, -1.0, False))
                    if letter == b'W' or letter == b'S':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, newstate, 0.0, False))

                    # if letter in b'GH':
                    #     li.append((1.0, s, 0, True))
                    # else:
                    #     if is_slippery:
                    #         for b in [(a-1)%4, a, (a+1)%4]:
                    #             newrow, newcol = inc(row, col, b)
                    #             newstate = to_s(newrow, newcol)
                    #             newletter = desc[newrow, newcol]
                    #             done = bytes(newletter) in b'GH'
                    #             rew = float(newletter == b'G')
                    #             li.append((1.0/3.0, newstate, rew, done))
                    #     else:
                    #         newrow, newcol = inc(row, col, a)
                    #         newstate = to_s(newrow, newcol)
                    #         newletter = desc[newrow, newcol]
                    #         done = bytes(newletter) in b'GH'
                    #         rew = float(newletter == b'G')
                    #         li.append((1.0, newstate, rew, done))

        super(MazeEnv, self).__init__(nS, nA, P, isd)

    def reset(self):
        self.s = self.start
        return self.start

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
# env = MazeEnv(maze_file='maze/maze.png')

class WindyMazeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, maze_file, wind_prob):
        self.desc = desc = read_maze(maze_file)
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        self.wind_prob = wind_prob

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            prev_row = row
            prev_col = col
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            if self.desc[row, col] == b'B':
                row = prev_row
                col = prev_col
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)


                #start state
                letter = desc[row, col]
                if letter == b'S':
                    self.start = s

                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter == b'B':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1.0, to_s(row, col), -1.0, True))
                        pass
                    if letter == b'Y':
                        # newrow, newcol = inc(row, col, a)
                        # newstate = to_s(newrow, newcol)
                        # li.append((1, newstate, 0.0, False))

                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        li.append((1 - wind_prob, newstate, 0, False))

                        #going up
                        newrow, newcol = inc(row, col, 3)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        li.append((wind_prob, newstate, 0, False))


                    if letter == b'G':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, to_s(row, col), 1.0, True))
                    # if letter == b'T':
                    #     newrow, newcol = inc(row, col, a)
                    #     newstate = to_s(newrow, newcol)
                    #     newletter = desc[newrow, newcol]
                    #     li.append((1, newstate, 1.0, False))
                    if letter == b'R':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        li.append((1, to_s(row, col), -1.0, True))
                    if letter == b'W' or letter == b'S':
                        newrow, newcol = inc(row, col, a)
                        newstate = to_s(newrow, newcol)
                        li.append((1, newstate, 0.0, False))

                    # if letter in b'GH':
                    #     li.append((1.0, s, 0, True))
                    # else:
                    #     if is_slippery:
                    #         for b in [(a-1)%4, a, (a+1)%4]:
                    #             newrow, newcol = inc(row, col, b)
                    #             newstate = to_s(newrow, newcol)
                    #             newletter = desc[newrow, newcol]
                    #             done = bytes(newletter) in b'GH'
                    #             rew = float(newletter == b'G')
                    #             li.append((1.0/3.0, newstate, rew, done))
                    #     else:
                    #         newrow, newcol = inc(row, col, a)
                    #         newstate = to_s(newrow, newcol)
                    #         newletter = desc[newrow, newcol]
                    #         done = bytes(newletter) in b'GH'
                    #         rew = float(newletter == b'G')
                    #         li.append((1.0, newstate, rew, done))

        super(WindyMazeEnv, self).__init__(nS, nA, P, isd)

    def reset(self):
        self.s = self.start
        return self.start

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()