# from BaxterReacherv0 import *
from .FlipBit import *
from .Maze import *
from .Robotics import *
from .RobotSuite import *
from .Pacman import MsPacman

def make_empty_maze(h, w, max_steps = 32,reward = "sparse"):
    return Maze(np.ones((h, w), dtype=np.int), max_steps, entries=[(0, 0)], reward=reward)

def make_random_layout(h, w):
    """Adapted from https://rosettacode.org/wiki/Maze_generation."""
    maze_string = ''

    vis = [[0] * w + [1] for _ in range(h)] + [[1] * (w + 1)]
    ver = [["| "] * w + ['|'] for _ in range(h)] + [[]]
    hor = [["+-"] * w + ['+'] for _ in range(h + 1)]

    def walk(x, y):
        vis[y][x] = 1

        d = [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)]
        np.random.shuffle(d)
        for (xx, yy) in d:
            if vis[yy][xx]:
                continue
            if xx == x:
                hor[max(y, yy)][x] = "+ "
            if yy == y:
                ver[y][max(x, xx)] = "  "
            walk(xx, yy)

    walk(np.random.randint(w), np.random.randint(h))
    for (a, b) in zip(hor, ver):
        maze_string += ''.join(a + ['\n'] + b) + '\n'

    A = [[]]
    for c in maze_string[: -2]:
        if c == '\n':
            A.append([])
        elif c == ' ':
            A[-1].append(1)
        else:
            A[-1].append(0)

    return np.array(A, dtype=np.int)


def make_random_maze(h, w, max_steps = 32,reward = "sparse"):
    return Maze(make_random_layout(h, w), max_steps, [(1, 1)], reward=reward)


def make_tmaze(length,max_steps = 32, reward = "sparse"):
    layout = np.zeros(shape=(3, length+1), dtype=np.int)

    layout[:, 0] = 1
    layout[1, :] = 1
    layout[:, -1] = 1

    return Maze(layout, max_steps, [(0, 0)], reward = reward)


def make_cheese_maze(length,max_steps = 32, reward = "sparse"):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.ones(shape=(length, 5), dtype=np.int)

    layout[1:, 1] = 0
    layout[1:, 3] = 0

    return Maze(layout, max_steps, [(length - 1, 2)], reward = reward)


def make_wine_maze(max_steps = 32, reward = "sparse"):
    """Adapted from Bakker, Pieter Bram. The state of mind. 2004, pg. 155"""
    layout = np.array([[0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 0, 1, 0, 1, 0],
                       [1, 1, 0, 1, 0, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0]], dtype=np.int)

    return Maze(layout, max_steps, [(1, 0)], reward = reward)


def make_four_rooms_maze(max_steps=32, reward = "sparse"):
    """Adapted from Sutton et al. Between MDPs and semi-MDPs: ... (1999)"""
    layout = np.ones(shape=(11, 11), dtype=np.int)

    # Walls
    layout[:, 5] = 0
    layout[5, :5] = 0
    layout[6, 6:] = 0

    # Doors
    layout[5, 1] = 1
    layout[2, 5] = 1
    layout[6, 8] = 1
    layout[9, 5] = 1

    # Average distance to exit is close to 10
    entries = [(0, 0), (0, 10), (10, 0), (10, 10)]

    return Maze(layout, max_steps, entries, epsilon=0.2, reward = reward)

def make_env(env_string, max_steps = 32, reward = "sparse"):
    env = None

    match = re.match('FlipBit(\d+)', env_string)
    if match:
        env = FlipBit(n_bits=int(match.group(1)), reward = reward)

    match = re.match('EmptyMaze(\d+)_(\d+)', env_string)
    if match:
        env = make_empty_maze(h=int(match.group(1)), w=int(match.group(2)),
                              max_steps=max_steps, reward = reward)

    match = re.match('RandomMaze(\d+)_(\d+)', env_string)
    if match:
        env = make_random_maze(h=int(match.group(1)), w=int(match.group(2)),
                               max_steps=max_steps, reward = reward)

    match = re.match('TMaze(\d+)', env_string)
    if match:
        env = make_tmaze(length=int(match.group(1)), max_steps=max_steps, reward = reward)

    match = re.match('CheeseMaze(\d+)', env_string)
    if match:
        env = make_cheese_maze(length=int(match.group(1)), max_steps=max_steps, reward = reward)

    match = re.match('WineMaze', env_string)
    if match:
        env = make_wine_maze(max_steps=max_steps, reward = reward)

    match = re.match('FourRoomMaze', env_string)
    if match:
        env = make_four_rooms_maze(max_steps=max_steps, reward = reward)


    match = re.match('FetchReachDiscrete', env_string)
    if match:
        env = FetchReachDiscrete(reward=reward)


    match = re.match('FetchPushDiscrete', env_string)
    if match:
        env = FetchPushDiscrete(reward=reward)


    match = re.match('FetchSlideDiscrete', env_string)
    if match:
        env = FetchSlideDiscrete(reward=reward)


    match = re.match("MsPacman",env_string)
    if match:
        env = MsPacman(reward = reward)

    if env is None:
        raise Exception('Invalid environment string.')

    return env