"""Run training with pytorch-lightning
below code is inspired by https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/reinforce-learning-DQN.html
"""
import argparse
import datetime
import itertools
import logging
import os
import ast
from collections import deque, namedtuple
from copy import deepcopy
from multiprocessing.sharedctypes import Value
from typing import Iterator, List, Tuple

import gymnasium as gym
import numpy as np
import room_env
import torch
import torch.optim as optim
import yaml
from gymnasium import spaces
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import IterableDataset

from env_multimem.room_env import RoomEnv1

from utils import write_json

logger = logging.getLogger()
logger.disabled = True

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state", "label"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them."""

    def __init__(self, replay_size: int) -> None:
        """

        Args
        ----
        replay_size: size of the buffer

        """
        self.buffer = deque(maxlen=replay_size)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args
        ----
        experience: tuple (state, action, reward, done, new_state)

        """
        self.buffer.append(experience)

    def sample(
        self, sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample data from the replay buffer."""
        indices = np.random.choice(len(self.buffer), sample_size, replace=False)
        states, actions, rewards, dones, next_states, labels = zip(
            *(self.buffer[idx] for idx in indices)
        )
        # print(labels)
        return states, actions, rewards, dones, next_states, labels


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated with new
    experiences during training.

    This allows us to train our DQN agent as if we do supervised learning.

    """

    def __init__(self, replay_buffer: ReplayBuffer, sample_size: int = 200) -> None:
        """

        Args
        ----
        buffer: replay buffer
        sample_size: number of experiences to sample at a time

        """
        self.replay_buffer = replay_buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states, labels = self.replay_buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i], labels[i]


class MockDataset(Dataset):
    """This is a mock dataset that has one data sample.

    Since pytorch lightning is made for supervised learning, not for RL,
    we do this hack."""

    def __init__(self) -> None:
        pass

    def __len__(self) -> None:
        return 1

    def __getitem__(self, idx) -> torch.Tensor:

        return torch.tensor([0], dtype=torch.float32)


class RLAgent:
    """RL Agent class handeling the interaction with the environment."""

    def __init__(
        self,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
        classification_buffer: ReplayBuffer,
        capacity: dict,
        pretrain_semantic: bool,
        policies: dict,
        steps_until_use_model_answer:None,
    ) -> None:
        """
        Args
        ----
        env: training environment
        replay_buffer: replay buffer storing experiences
        capacity: memory system capacity
            e.g., {"episodic":1, "semantic": 1, "short": 1}
        pretrain_semantic: Pretrain the semantic memory system from ConceptNet.
        policies:
            e.g., {
                "memory_management": "rl",
                "question_answer": "episodic_semantic",
                "encoding": "argmax",
            }
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.classification_buffer = classification_buffer
        self.capacity = capacity
        self.pretrain_semantic = pretrain_semantic
        self.policies = policies
        self.create_spaces()
        self.debug_dump = []
        self.question = tuple([1, 2, 3, 4, 5])

        # NOTE: This toggles the difficulty of our method 
        # self.pass_in_answer = False
        self.steps_until_use_model_answer = steps_until_use_model_answer

        self.reset()

    def create_spaces(self) -> None:
        """Create action space. The size is 3."""
        if self.policies["memory_management"].lower() == "rl":
            n_actions = 3
        elif self.policies["question_answer"].lower() == "rl":
            n_actions = 2
        else:
            raise NotImplementedError
        self.action_space = spaces.Discrete(n_actions)

    def reset(self) -> None:
        """Reset the environment and update the state."""
        self.debug_dump = []
        self.state, info = self.env.reset()
        self.question = tuple([1, 2, 3, 4, 5])

        # The lists will be converted to str temporarily ...
        self.state = [
            str(self.state["episodic"]),
            str(self.state["semantic"]),
            str(self.state["short"]),
            str(tuple([1, 2, 3, 4, 5]))
        ]


    def get_action(
        self,
        net: nn.Module,
        epsilon: float,
        step: int,
        use_model_action: bool
    ) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.
        Args
        ----
        net: DQN network
        epsilon: value to determine likelihood of taking a random action
        step
        Returns
        -------
        action
        """
        actions = {}
        if np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            q_values, _ = net(self.state[:3])
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

            to_dump = {
                "state": self.state,
                "q_values": q_values.detach().cpu().tolist(),
                "step": step,
            }
            self.debug_dump.append(to_dump)
        
        # Include the question to have output a filter and answer

        # TODO: Allow to take out the filter as well
        answer, _ = net(self.state)
        # print("answer output: ", answer)
        answer = torch.argmax(answer)

        actions["memory_management_action"] = action 


        actions["answer_action"] = int(answer.item()) if use_model_action else None 
        # print("Answer action in train.py: ", actions["answer_action"], answer.item(), self.pass_in_answer, self.state[0])
        # print("chosen action: ", actions["answer_action"])
        # print("answer_action in train.py: ", actions["answer_action"])

        return actions

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        step: int = None,
        use_model_action: bool = False,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.
        action=0 will put the oldest short memory into the episodic
        action=1 will put it into the semantic.
        action=2 will throw it away.
        Args
        ----
        net: DQN network
        epsilon: value to determine likelihood of taking a random action
        step: step number for the RL agent.
        Returns
        -------
        reward, done
        """
        actions = self.get_action(net, epsilon, step, use_model_action)

        action = actions["memory_management_action"]

        # info["correct_filter"] and info["correct_answer"] and info["next_question"] are filled
        new_state, reward, done, truncated, info = self.env.step(actions)

        # The lists will be converted to str temporarily ...
        next_question = info["next_question"] if info["next_question"] != None else tuple([1, 2, 3, 4, 5])
        new_state = [
            str(new_state["episodic"]),
            str(new_state["semantic"]),
            str(new_state["short"]),
            str(next_question)
        ]

        exp = Experience(deepcopy(self.state), action, reward, done, new_state, info["correct_answer"])

        self.question = next_question

        self.replay_buffer.append(exp)


        self.state = new_state
        if done:
            self.debug_dump_final = deepcopy(self.debug_dump)
            self.reset()

        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 32,
        lr: float = 1e-2,
        gamma: float = 0.5,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_step: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        epoch_length: int = 200,
        des_size: str = "xxs",
        seed: int = 42,
        capacity: dict = {"episodic": 1, "semantic": 1, "short": 1},
        policies: dict = {
            "memory_management": "rl",
            "question_answer": "episodic_semantic",
            "encoding": "argmax",
        },
        question_prob: float = 0.1,
        observation_params: str = "pefect",
        nn_params: dict = None,
        allow_random_human: bool = False,
        allow_random_question: bool = False,
        loss_function: str = "Huber",
        optimizer: str = "Adam",
        pretrain_semantic: bool = False,
        num_eval_iter: int = 5,
        varying_rewards: bool = False,
        accelerator: str = "cpu",
        use_filter: bool = False,
        **kwargs,
    ) -> None:
        """

        Args
        ----
        batch_size: number of data samples in one batch.
        lr: learning rate
        gamma: discount factor
        sync_rate: how many frames do we update the target network
        replay_size: capacity of the replay buffer
        warm_start_size: how many samples do we use to fill our buffer at the start of
            training
        eps_last_step: what frame should epsilon stop decaying
        eps_start: starting value of epsilon
        eps_end: final value of epsilon
        epoch_length: max length of an episode
        des_size: "xxs", "xs", "s", "m", or "l".
        seed: random seed
        capacity: memory system capacity
            e.g., {"episodic":1, "semantic": 1, "short": 1}
        policies:
            e.g., {
                "memory_management": "rl",
                "question_answer": "episodic_semantic",
                "encoding": "argmax",
            }
        question_prob: The probability of a question being asked at every observation.
        observation_params: At the moment this is only "perfect".
        nn_params: neural network parameters
            e.g.,
                architecture: lstm
                embedding_dim: 8
                hidden_size: 16
                num_layers: 2
        allow_random_human: whether to allow the random human sampling
        allow_random_question: whether to allow the random question sampling
        loss_function: either Huber or MSE
            Huber punishes outliers less than MSE
        optimizer: either adam or rmsprop
        pretrain_semantic: Whether or not to pretrain the semantic memory system
            from ConceptNet.
        num_eval_iter: number of iterations for evaluation.
        accelerator: "cpu", "gpu", or "auto"

        """
        super().__init__()
        self.save_hyperparameters()

        self.env = RoomEnv1(
                des_size=self.hparams.des_size,
                seed=self.hparams.seed,
                policies=self.hparams.policies,
                capacity=self.hparams.capacity,
                question_prob=self.hparams.question_prob,
                observation_params=self.hparams.observation_params,
                allow_random_human=self.hparams.allow_random_human,
                allow_random_question=self.hparams.allow_random_question,
                pretrain_semantic=self.hparams.pretrain_semantic,
                check_resources=False,
                varying_rewards=self.hparams.varying_rewards,
        )

        self.replay_buffer = ReplayBuffer(self.hparams.replay_size)
        self.classification_buffer = ReplayBuffer(self.hparams.replay_size)
        self.steps_until_use_model_answer = 0 # None 
        self.num_steps_so_far = 0
        self.use_model_action = False

        self.agent = RLAgent(
            env=self.env,
            replay_buffer=self.replay_buffer,
            classification_buffer=self.classification_buffer,
            capacity=self.hparams.capacity,
            pretrain_semantic=self.hparams.pretrain_semantic,
            policies=self.hparams.policies,
            steps_until_use_model_answer=self.steps_until_use_model_answer
        )

        if self.hparams.nn_params["architecture"].lower() == "lstm":
            from model import LSTM as DQN
        else:
            raise NotImplementedError

        self.hparams.nn_params["n_actions"] = self.agent.action_space.n
        # TODO: Change this to account for the code changes in environment (first_human...etc.)
        self.hparams.nn_params["entities"] = {
            "people": self.env.des.people,
            "objects": self.env.des.objects,
            "small_locations": self.env.des.small_locations,
            "big_locations": self.env.des.big_locations,
            "relations": self.env.des.relations,
        }
        self.hparams.nn_params["capacity"] = self.hparams.capacity
        self.hparams.nn_params["accelerator"] = self.hparams["accelerator"]
        self.hparams.nn_params["use_filter"] = self.hparams["use_filter"]

        self.use_filter = self.hparams["use_filter"]
        self.filter_reg = self.hparams["filter_regularization"]
        self.training_offset = self.hparams["training_offset"]
        self.classification_batch = self.hparams["classification_batch"]
        self.capacity = self.hparams.capacity

        self.net = DQN(**self.hparams.nn_params)
        self.target_net = DQN(**self.hparams.nn_params)

        self.class_optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)# self.hparams.lr)

        if self.hparams["accelerator"] == "gpu":
            self.net.to("cuda")
            self.target_net.to("cuda")

        self.target_net.eval()

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_loss = []
        self.populate(self.hparams.warm_start_size)
        self.env.reset()
        self.num_validations = 0
        self.automatic_optimization = False

    def populate(self, warm_start_size: int = 1000) -> None:
        """Carry out several random steps through the environment to initially fill
        up the replay buffer with experiences.

        Args
        ----
        warm_start_size: number of random steps to populate the buffer with

        """
        for _ in range(warm_start_size):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Pass in a state x through the network and gets the q_values of each action
        as an output.

        Args
        ----
        x: environment state

        Returns
        -------
        q values

        """
        output, _ = self.net(x)
        return output

    def td_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculate the TD loss (either huber or mse) using a mini batch from the
        replay buffer.

        Args
        ----
        batch: current mini batch of replay data

        Returns
        -------
        mse or huber batch loss

        """
        states, actions, rewards, dones, next_states, _ = batch
        # print("td loss states ", states)
        state_action_values = (
            self.net(states[:3])[0].gather(1, actions.long().unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states[:3])[0].max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        if self.hparams.loss_function.lower() == "mse":
            return nn.MSELoss()(state_action_values, expected_state_action_values)
        elif self.hparams.loss_function.lower() == "huber":
            return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)
        else:
            raise ValueError

    def classification_loss(self, batch:[Tensor, Tensor]) -> Tensor:
        states, _, _, _, _, labels = batch

        # Take out all of the states that have a None value for its question 
        # print("Was any None: ", not all([ast.literal_eval(state) != (1, 2, 3, 4, 5) for i, state in enumerate(states[3])]))
        idx = [i for i, state in enumerate(states[3]) if ast.literal_eval(state) != (1, 2, 3, 4, 5)]
        # print("Filtered successfully: ", not np.all([ast.literal_eval(state) != (1, 2, 3, 4, 5) for state in states[3]]))
        states = [tuple([s for i, s in enumerate(list(state)) if i in idx]) for state in states]
        # rewards = [r for i, r in enumerate(rewards) if i in idx]
        labels = [l for i, l in enumerate(labels) if i in idx]

        probs, memory_filter_probs = self.net(states)

        loss = torch.nn.CrossEntropyLoss()

        targets = torch.LongTensor(labels).to(self.device)
        # print(probs.size(), targets.size())
        answer_loss = loss(probs, targets)

        if self.use_filter:
            # regularization_loss = self.filter_reg * torch.mean(torch.sum(torch.square(memory_filter), dim=1))
            # print("Regularization loss: ", regularization_loss, "    Approximate num memories on average: ", regularization_loss / self.filter_reg, "    Std num memories: ", torch.std(torch.sum(torch.abs(memory_filter), dim=1)))
            # answer_loss += regularization_loss
            log_probs = torch.sum(torch.log(memory_filter_probs), dim=1)

            # Action probs
            preds = torch.argmax(probs, dim=1).detach()

            norm_probs = torch.nn.functional.softmax(probs, dim=1)
            norm_probs, preds = torch.max(norm_probs, dim=1)

            preds = torch.where(norm_probs > .5, preds, -1)  # -1 is used because there is no prediction with value -1

            # Define rewards based on the current model's prediction (not the past one)
            rewards = torch.where(preds == targets, 1.0, -1.0)

            filter_loss = -torch.mean(log_probs * rewards)  # TODO: Instead of using rewards here, use whether or not the argmax of probs is the same as label. That way, it's still on-policy

            regularization_loss = torch.mean(torch.sum(memory_filter_probs, dim=1))

            print("Policy loss:    ", filter_loss, "    Regularization loss: ", self.filter_reg * regularization_loss / sum(self.capacity.values()), "  Avg num memories: ", regularization_loss)
            filter_loss = (filter_loss) + self.filter_reg * regularization_loss / sum(self.capacity.values())

            answer_loss += filter_loss

        # print("classificaiton loss: ", answer_loss)
        return answer_loss

    def get_epsilon(self, start: int, end: int, eps_last_step: int) -> float:
        """Get the epsilon value. The scheduling is done lienarly.

        Consider exponential decay, although this wouldn't change so much.

        Args
        ----
        start: eps start value (e.g., 1.0)
        end: eps end value (e.g., 0)
        eps_last_step: The time step after which the eps value doesn't decrease anymore.

        """
        if self.global_step > eps_last_step:
            return end
        return start - (self.global_step / eps_last_step) * (start - end)

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch: int) -> float:
        """Carry out a single step through the environment to update the replay
        buffer. Then calculates loss based on the minibatch recieved.
        Args
        ----
        batch: current mini batch of replay data
        nb_batch: batch number
        Returns
        -------
        Training loss and log metrics
        """
        # print("state input: ", batch[0])
        epsilon = self.get_epsilon(
            self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_step
        )
        self.log("epsilon", epsilon)

        # step through environment with agent
        reward, done = self.agent.play_step(
            self.net,
            epsilon,
            step=self.global_step,
            use_model_action = self.use_model_action
        )
        self.episode_reward += reward
        self.log(
            "episode_reward", torch.tensor(self.episode_reward, dtype=torch.float32)
        )

        if self.steps_until_use_model_answer != None and self.num_steps_so_far >= self.steps_until_use_model_answer:
            if self.num_steps_so_far == self.steps_until_use_model_answer:
                epochs = 2
                print('\nBeginning training for {} epochs'.format(epochs), flush=True)
                sample = len(self.replay_buffer)
            else:
                epochs = 1
                sample = self.classification_batch * self.training_offset 
            for _ in range(epochs):
                class_loss = 0
                dataset = RLDataset(self.replay_buffer, sample)
                dataloader = DataLoader(dataset=dataset, batch_size=sample)
                counts = 0 
                for batch_classification in dataloader:
                    self.class_optimizer.zero_grad()
                    classification_loss = self.classification_loss(batch_classification)
                    class_loss += classification_loss
                    classification_loss.backward()
                    # self.manual_backward(classification_loss)
                    self.class_optimizer.step()
                    counts += 1
                    if counts >= self.training_offset:
                        break
                print("\n\nMean classification loss: ", class_loss / self.training_offset, flush=True)

        opt = self.optimizers() 
        # calculates training loss from the given batch
        opt.zero_grad()
        loss = self.td_loss(batch)
        self.episode_loss.append(loss)
        self.log("loss", loss)
        print("TD Loss: ", loss, "   Steps so far: ", self.num_steps_so_far, "   Using model action? ", self.use_model_action)
        self.manual_backward(loss)
        opt.step()


        if done:
            self.log(
                "total_reward", torch.tensor(self.episode_reward, dtype=torch.float32)
            )
            self.episode_reward = 0

            self.log(
                "train_episode_average_loss",
                torch.mean(
                    torch.tensor(self.episode_loss, dtype=torch.float32),
                    dtype=torch.float32,
                ),
            )
            self.episode_loss = []

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.num_steps_so_far += 1
        if self.steps_until_use_model_answer != None and self.num_steps_so_far > self.steps_until_use_model_answer:
            self.use_model_action = True

        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], nb_batch: int) -> None:
        """Run validation.

        This does not involve using pytorch dataset / dataloaders. It's a hack.
        """
        self.agent.reset()
        val_epsilon = 0
        val_episode_reward = []
        to_dump = {}
        for idx in range(self.hparams.num_eval_iter):
            val_episode_reward_ = 0
            for step in itertools.count():
                # global step doesn't increase here, since no backprop is happening.
                reward, done = self.agent.play_step(
                    self.net,
                    val_epsilon,
                    step=step,
                    use_model_action=self.use_model_action
                )
                val_episode_reward_ += reward
                if done:
                    break
            val_episode_reward.append(val_episode_reward_)
            to_dump[idx] = deepcopy(self.agent.debug_dump_final)

        val_episode_reward_mean = round(np.mean(val_episode_reward).item(), 3)
        val_episode_reward_std = round(np.std(val_episode_reward).item(), 3)

        self.log(
            "val_total_reward_mean",
            torch.tensor(val_episode_reward_mean, dtype=torch.float32),
        )
        self.log(
            "val_total_reward_std",
            torch.tensor(val_episode_reward_std, dtype=torch.float32),
        )
        debug_path = os.path.join(
            self.logger.log_dir,
            f"val_debug_episode={self.num_validations}-mean={val_episode_reward_mean}"
            f"-std={val_episode_reward_std}.json",
        )

        write_json(to_dump, debug_path)
        self.num_validations += 1

    def test_step(self, batch: Tuple[Tensor, Tensor], nb_batch: int) -> None:
        """Run test.

        This does not involve using pytorch dataset / dataloaders. It's a hack.
        """
        self.agent.reset()
        test_epsilon = 0
        test_episode_reward = []
        to_dump = {}
        for idx in range(self.hparams.num_eval_iter):
            test_episode_reward_ = 0
            for step in itertools.count():
                reward, done = self.agent.play_step(
                    self.net,
                    test_epsilon,
                    step=step,
                    use_model_action=self.use_model_action
                )
                test_episode_reward_ += reward
                if done:
                    break
            test_episode_reward.append(test_episode_reward_)
            to_dump[idx] = deepcopy(self.agent.debug_dump_final)

        test_episode_reward_mean = round(np.mean(test_episode_reward).item(), 3)
        test_episode_reward_std = round(np.std(test_episode_reward).item(), 3)

        self.log(
            "test_total_reward_mean",
            torch.tensor(test_episode_reward_mean, dtype=torch.float32),
        )
        self.log(
            "test_episode_reward_std",
            torch.tensor(test_episode_reward_std, dtype=torch.float32),
        )
        debug_path = os.path.join(
            self.logger.log_dir,
            f"test_debug-mean={test_episode_reward_mean}-std={test_episode_reward_std}"
            ".json",
        )

        write_json(to_dump, debug_path)

    def configure_optimizers(self) -> List[optim.Optimizer]:
        """Initialize optimizer."""

        if self.hparams.optimizer.lower() == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        elif self.hparams.optimizer.lower() == "rmsprop":
            optimizer = optim.RMSprop(self.net.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError

        return {"optimizer": optimizer}

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.replay_buffer, self.hparams.epoch_length)
        # num_workers > 1 will have a weird effect.
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams.batch_size)

        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """A mock dataloader."""
        dataset = MockDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        return dataloader

    def test_dataloader(self) -> DataLoader:
        """A mock dataloader."""
        dataset = MockDataset()
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        return dataloader


def main(**kwargs):
    """Make pytorch lightning objects and start training."""

    # The room env has 128 steps fixed as one episode.
    assert (
        kwargs["epoch_length"] / kwargs["batch_size"] == 128
    ), "one epoch should be one episode!"

    model = DQNLightning(**kwargs)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_total_reward_mean",
        mode="max",
        filename="{epoch:02d}-{val_total_reward_mean:.2f}-{val_total_reward_std:.2f}",
    )
    # train_end_callback = TrainEndCallback()
    early_stop_callback = EarlyStopping(
        monitor="val_total_reward_mean",
        strict=False,
        min_delta=0.00,
        patience=kwargs["early_stopping_patience"],
        verbose=True,
        mode="max",
    )

    trainer = Trainer(
        accelerator=kwargs["accelerator"],
        devices="auto",
        max_epochs=kwargs["max_epochs"],
        precision=kwargs["precision"],
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=kwargs["log_every_n_steps"],
        num_sanity_val_steps=0,
        default_root_dir=f"./training_results/{str(datetime.datetime.now())}",
    )
    trainer.fit(model)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument(
        "-c",
        "--config",
        default="./train.yaml",
        type=str,
        help="config file path (default: ./train.yaml)",
    )
    args = parser.parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
