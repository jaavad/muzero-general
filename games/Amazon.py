import datetime
import math
import os

import numpy
import torch

from .abstract_game import AbstractGame

#
#
#   This representation of amazon uses the one move representation, which means the moving and the shooting are done simultaneously
#
#


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
        self.max_num_gpus = None

        # Size of action space is calculated as follows:
        # --36 locations to choose piece from.
        # --select 1 out of 8 directions to move
        # --select a distance to move, 1 to 5
        # --select 1 out of 8 directions to shoot
        # --select a distance to shoot, 1 to 5
        #   which sums to a total of 36*40*40

        # Game
        # Dimensions of the game observation, must be 3 (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.observation_shape = (4, 6, 6)
        # Fixed list of all possible actions. You should only edit the length
        self.action_space = list(range(36*40*40))
        # List of players. You should only edit the length
        self.players = list(range(2))
        # Number of previous observations and previous actions to add to the current observation
        self.stacked_observations = 0

        # Evaluate
        # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.muzero_player = 0
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        # Self-Play
        # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.num_workers = 2
        self.selfplay_on_gpu = False
        self.max_moves = 33  # Maximum number of moves if game is not finished before
        self.num_simulations = 400  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
        self.temperature_threshold = None

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 10

        # Residual Network
        # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.downsample = False
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [64]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [64]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [64]

        # Fully Connected Network
        self.encoding_size = 32
        # Define the hidden layers in the representation network
        self.fc_representation_layers = []
        # Define the hidden layers in the dynamics network
        self.fc_dynamics_layers = [64]
        # Define the hidden layers in the reward network
        self.fc_reward_layers = [64]
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        # Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[
                                         :-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        # Total number of training steps (ie weights update according to a batch)
        self.training_steps = 100000
        self.batch_size = 512  # Number of parts of games to train on at each training step
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = 50
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000

        # Replay Buffer
        # Number of self-play games to keep in the replay buffer
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 33  # Number of game moves to keep for every batch element
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 33
        # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 0.8

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = True

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Amazon()

    def step(self, action):
        """
        Apply action to the game.
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action)


def queenMove(x, y, dir, dist):
    tmp = [(-1, 0), (-1, 1), (0, 1), (1, 1),
           (1, 0), (1, -1), (0, -1), (-1, -1)]
    return (x+tmp[dir][0]*dist, y+tmp[dir][1]*dist)


def linear_to_sextuple(action):
    x = (action % 6)
    y = (action % 36) // 6
    move_dir = (action % 288) // 36
    move_len = ((action % 1440) // 288)+1
    shoot_dir = (action % 11520) // 1440
    shoot_len = (action // 11520)+1
    return (shoot_len, shoot_dir, move_len, move_dir, y, x)


class Amazon:
    def __init__(self):
        self.board_size = 6
        self.board = numpy.zeros(
            (self.board_size, self.board_size, 2), dtype="int32")
        self.player = 1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    # returns a list of quadruples [(dir,len,xx,yy)]
    def get_legal_queen_moves(self, x, y):
        moves = []
        for i, dir in enumerate([(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]):
            for l in range(1, self.board_size):
                xx, yy = x + dir[0] * l, y + dir[1] * l
                if xx < 0 or xx >= self.board_size or yy < 0 or yy >= self.board_size or self.board[xx, yy, 0] != 0 or self.board[xx, yy, 1] != 0:
                    break
                else:
                    moves.append((i, l-1, xx, yy))
        return moves

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros(
            (self.board_size, self.board_size, 2), dtype="int32")
        self.board[0, 1, 0] = 1
        self.board[1, 5, 0] = 1
        self.board[4, 0, 0] = -1
        self.board[5, 4, 0] = -1
        self.player = 1
        return self.get_observation()

    def step(self, action):
        #x = (action % 6) // 1
        #y = (action % (6*6)) // 6
        #move_dir = (action % (6 * 6 * 8)) // (6*6)
        #move_len = (action % (6 * 6 * 8 * 5)) // (6*6*8)
        #shoot_dir = (action % (6*6*8*5*8)) // (6*6*8*5)
        #shoot_len = (action) // (6*6*8*5*8)
        x = (action % 6)
        y = (action % 36) // 6
        move_dir = (action % 288) // 36
        move_len = ((action % 1440) // 288)+1
        shoot_dir = (action % 11520) // 1440
        shoot_len = (action // 11520)+1

        self.board[x, y, 0] = 0
        xx, yy = queenMove(x, y, move_dir, move_len)
        self.board[xx, yy, 0] = self.player
        xxx, yyy = queenMove(xx, yy, shoot_dir, shoot_len)
        self.board[xxx, yyy, 1] = 1

        done = self.is_finished()

        reward = 1 if done else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board[:, :, 0] == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board[:, :, 0] == -1, 1.0, 0.0)
        board_arrows = self.board[:, :, 1]
        board_to_play = numpy.full((6, 6), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_arrows, board_to_play])

    def legal_actions(self):
        legal = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.board[x, y, 0] == self.player:
                    legalmoves = self.get_legal_queen_moves(x, y)
                    for move_dir, move_len, xx, yy in legalmoves:
                        # legal shoots at this moment doesn't include shooting back to the square
                        # where it came from, so need to manually add that in
                        legalshoots = self.get_legal_queen_moves(xx, yy)
                        for shoot_dir, shoot_len, _, _ in legalshoots:
                            legal.append(shoot_len*11520+shoot_dir *
                                         1440+move_len*288+move_dir*36+y*6+x)
                        # shooting back to the location where it came from means the direction
                        # is rotated by 180 degrees and the distance is kept constant
                        legal.append(move_len*11520+((move_dir+4) %
                                     8)*1440+move_len*288+move_dir*36+y*6+x)
        return legal

    def is_finished(self):
        return len(self.legal_actions()) == 0

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                if self.board[row, col, 0] == 1:
                    print("O", end=" ")
                elif self.board[row, col, 0] == -1:
                    print("X", end=" ")
                elif self.board[row, col, 1] == 1:
                    print("*", end=" ")
                else:
                    print(".", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y

#import games.amazon_rep1
#game = games.amazon_rep1.Amazon()