"""Room environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py), and Memory classes.
This is a more generalized version than RoomEnv0.
"""
import logging
import os
import random
from copy import deepcopy
from typing import Tuple

import gymnasium as gym

from .des import RoomDes
from .memory import EpisodicMemory, SemanticMemory, ShortMemory
from .policy import answer_question, encode_observation, manage_memory
from .utils import seed_everything
from .answer import Answer

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class RoomEnv1(gym.Env):
    """The Room environment version 1.

    This env includes three state-action spaces. You have to choose which one of the
    three will be RL trained.

    Memory management.
        State: episodic, semantic, and short-term memory systems at time t
        Action: (0) Move the oldest short-term memory to the episodic,
                (1) to the semantic, or (2) forget it

    Question-answer
        State: episodic and semantic memory systems at time t
        Action: (0) Select the episodic memory system to answer the question, or
                (1) the semantic

    Encoding an observation to a short-term memory. The state space is
        (i) triple-based, (ii) text-based, or (iii) image-based.
        Triple
            State: [(head_i, relation_i, tail_i) | i is from 1 to N]
            Action: Choose one of the N triples (actions) to be encoded as
                    a short-term memory.
        Text
            State: [token_1, token_2, …, token_N]
            Action: This is actually now N^3, where the first, second and third are to
                    choose head, relation, and tail, respectively.
        Image
            State: An image with objects
            Action: Not sure yet …

    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        des_size: str = "l",
        seed: int = 42,
        policies: dict = {
            "memory_management": "RL",
            "question_answer": "episodic_semantic",
            "encoding": "argmax",
        },
        capacity: dict = {"episodic": 16, "semantic": 16, "short": 1},
        question_prob: int = 1.0,
        observation_params: str = "perfect",
        allow_random_human: bool = False,
        allow_random_question: bool = False,
        total_episode_rewards: int = 128,
        pretrain_semantic: bool = False,
        check_resources: bool = True,
        varying_rewards: bool = False,
    ) -> None:
        """

        Args
        ----
        des_size: "xxs", "xs", "s", "m", or "l".
        seed: random seed number
        policies:
            memory_management:
                "RL": Reinforcement learning to learn the policy.
                "episodic": Always take action 1: move to the episodic.
                "semantic": Always take action 2: move to the semantic.
                "forget": Always take action 3: forget the oldest short-term memory.
                "random": Take one of the three actions uniform-randomly.
                "neural": Neural network policy
            question_answer:
                "RL": Reinforcement learning to learn the policy.
                "episodic_semantic": First look up the episodic and then the semantic.
                "semantic_episodic": First look up the semantic and then the episodic.
                "episodic": Only look up the episodic.
                "semantic": Only look up the semantic.
                "random": Take one of the two actions uniform-randomly.
                "neural": Neural network policy
            encoding:
                "RL": Reinforcement learning to learn the policy.
                "argmax": Take the triple with the highest score.
                "neural": Neural network policy
        capacity: memory capactiy of the agent.
            e.g., {"episodic": 1, "semantic": 1}
        question_prob: The probability of a question being asked at every observation.
        observation_params: At the moment this is only "perfect".
        allow_random_human: whether or not to generate a random human sequence.
        allow_random_question: whether or not to geneate a random question sequence.
        total_episode_rewards: total episode rewards
        pretrain_semantic: whether to prepopulate the semantic memory with ConceptNet
                           or not
        check_resources: whether to check the resources in the DES.
        varying_rewards: If true, then the rewards are scaled in every episode so that
             total_episode_rewards is total_episode_rewards.

        """
        self.seed = seed
        seed_everything(self.seed)
        self.policies = policies
        assert len([pol for pol in self.policies.values() if pol.lower() == "rl"]) == 1
        self.capacity = capacity
        self.question_prob = question_prob

        self.observation_params = observation_params

        self.allow_random_human = allow_random_human
        self.allow_random_question = allow_random_question
        self.total_episode_rewards = total_episode_rewards
        self.pretrain_semantic = pretrain_semantic
        self.check_resources = check_resources
        self.varying_rewards = varying_rewards

        # Our state space is quite complex. Here we just make a dummy observation space.
        # to bypass the sanity check.
        self.observation_space = gym.spaces.Discrete(1)

        if self.policies["memory_management"].lower() == "rl":
            # 0 for episodic, 1 for semantic, and 2 to forget
            self.action_space = gym.spaces.Discrete(3)
        if self.policies["question_answer"].lower() == "rl":
            # 0 for episodic and 1 for semantic
            self.action_space = gym.spaces.Discrete(2)
        if self.policies["encoding"].lower() == "rl":
            raise NotImplementedError

        self.des_size = des_size
        self.des = RoomDes(
            des_size=self.des_size,
            check_resources=self.check_resources,
        )
        assert 0 < self.question_prob <= 1

        self.init_memory_systems()
        self.answer_generator = Answer()

    # NOTE: This is unchanged
    def init_memory_systems(self) -> None:
        """Initialize the agent's memory systems."""
        self.memory_systems = {
            "episodic": EpisodicMemory(capacity=self.capacity["episodic"]),
            "semantic": SemanticMemory(capacity=self.capacity["semantic"]),
            "short": ShortMemory(capacity=self.capacity["short"]),
        }

        self.ground_truth_memory_systems = {
            "episodic": EpisodicMemory(capacity=1000000), 
            "semantic": SemanticMemory(capacity=10),
            "short": ShortMemory(capacity=1)
        }

        if self.pretrain_semantic:
            assert self.capacity["semantic"] > 0
            _ = self.memory_systems["semantic"].pretrain_semantic(
                self.des.semantic_knowledge,
                return_remaining_space=False,
                freeze=False,
            )

    # NOTE:
    def generate_sequences(self) -> None:
        """Generate human and question sequences in advance."""
        if self.observation_params.lower() == "perfect":
            if self.allow_random_human:
                self.human_sequence = random.choices(
                    list(self.des.first_humans), k=self.des.until + 1
                )
            else:
                self.human_sequence = (
                    self.des.first_humans * (self.des.until // len(self.des.first_humans) + 1)
                )[: self.des.until + 1]
        else:
            raise NotImplementedError

        if self.allow_random_question:
            # Questions are in the form of a quintuple as well 

            # For now, just sample all combinations of object-object, object-small location, small-location relations.
            # Might need to change so that we have more equal distribution of yes/no answers
            len_sequence = self.des.until + 1
            self.question_sequence = []  # questions are in the form (first_human, first_object, relation, second_human, second_object)?
            self.des.run()
            possible_objects = set()
            possible_small_locations = set()
            possible_big_locations = set()
            

            possible_questions = []

            self.human_sequence = []
            print("GENERATING HUMAN SEQUENCE")

            for i in range(len_sequence):
                state = self.des.states[i]
                human_candidates = []
                for h in state.keys():
                    if state[h]["relation"] == "AtLocation":
                        human_candidates.append(h)
                    else:  # NextTo
                        # Ensure that the second object has been seen 
                        second_human = state[h]["second_human"]
                        second_object = state[h]["second_object"]

                        if second_human != "Nature":  # it's an object
                            if (second_human, second_object) in possible_objects:
                                human_candidates.append(h)
                        else: # small_location
                            if (second_human, second_object) in possible_small_locations:
                                human_candidates.append(h)
            
                human = random.choice(human_candidates)
                self.human_sequence.append(human)
                # Each human represents the observation that is going to be given. Add 


                human = self.human_sequence[i]
                human_object = state[human]["first_object"]

                new_entities = {"object": None, "small_location": None, "big_location": None}

                if human == "Nature":
                    if (human, human_object) not in possible_small_locations: 
                        new_entities["small_location"] = (human, human_object)
                    possible_small_locations.add((human, human_object))
                    if state[human]["relation"] == "AtLocation":
                        if ("Nature", state[human]["second_object"]) not in possible_big_locations:
                            new_entities["big_location"] = ("Nature", state[human]["second_object"])
                        possible_big_locations.add(("Nature", state[human]["second_object"]))
                else:
                    if (human, human_object) not in possible_objects:
                        new_entities["object"] = (human, human_object)
                    possible_objects.add((human, human_object))

                if new_entities["object"]:
                    for o in possible_objects:
                        if o != new_entities["object"]:
                            possible_questions.append((o[0], o[1], "NextTo", new_entities["object"][0], new_entities["object"][1]))
                    for s in possible_small_locations:
                        possible_questions.append((new_entities["object"][0], new_entities["object"][1], "AtLocation", s[0], s[1]))
                if new_entities["small_location"]:
                    for o in possible_objects:
                        if o != new_entities["object"]:
                            possible_questions.append((o[0], o[1], "AtLocation", new_entities["small_location"][0], new_entities["small_location"][1]))
                    for s in possible_small_locations:
                        if s != new_entities["small_location"]:
                            possible_questions.append((s[0], s[1], "NextTo", new_entities["small_location"][0], new_entities["small_location"][1]))
                    for b in possible_big_locations:
                        possible_questions.append((new_entities["small_location"][0], new_entities["small_location"][1], "AtLocation", b[0], b[1]))
                if new_entities["big_location"]:
                    for s in possible_small_locations:
                        if s != new_entities["small_location"]:
                            possible_questions.append((s[0], s[1], "AtLocation", new_entities["big_location"][0], new_entities["big_location"][1]))

                
                if len(possible_questions) == 0:
                    self.question_sequence.append(None)
                else:
                    self.question_sequence.append(random.choice(possible_questions))

            assert len(possible_questions) == len(set(possible_questions))

            self.des._initialize()

        else:
            """
            self.question_sequence = [self.human_sequence[0]]
            self.des.run()
            assert (
                len(self.des.states)
                == len(self.des.events) + 1
                == len(self.human_sequence)
            )
            for i in range(len(self.human_sequence) - 1):  
                start = max(i + 2 - len(self.des.humans), 0) 
                end = i + 2
                humans_observed = self.human_sequence[start:end]

                current_state = self.des.states[end - 1]
                humans_not_changed = []
                for j, human in enumerate(humans_observed):
                    observed_state = self.des.states[start + j]

                    is_changed = False
                    for to_check in ["object", "object_location"]:
                        if (
                            current_state[human][to_check]
                            != observed_state[human][to_check]
                        ):
                            is_changed = True
                    if not is_changed:
                        humans_not_changed.append(human)

                self.question_sequence.append(random.choice(humans_not_changed)) 
                """

            self.des._initialize()

        effective_question_sequence = []
        for i, question in enumerate(self.question_sequence[:-1]):
            if random.random() < self.question_prob and question:  # make sure the question is not None
                effective_question_sequence.append(question)
            else:
                effective_question_sequence.append(None)
            
        # The last observation shouldn't have a question
        effective_question_sequence.append(None)
        self.question_sequence = effective_question_sequence

        assert len(self.human_sequence) == len(self.question_sequence)

        self.num_questions = sum(
            [True for question in self.question_sequence if question is not None]
        )
        if self.varying_rewards:
            self.CORRECT = self.total_episode_rewards / self.num_questions
            self.WRONG = -self.CORRECT
        else:
            self.CORRECT = 1
            self.WRONG = -1

    @staticmethod
    def extract_memory_entires(memory_systems: dict) -> dict:
        """Extract the entries from the Memory objects.
        Ars
        ---
        memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                        "short": ShortMemory}

        Returns
        -------
        memory_systems_: memory_systems only with entries.
        """
        memory_systems_ = {}
        for key, value in memory_systems.items():
            memory_systems_[key] = deepcopy(value.entries)

        return memory_systems_

    def generate_oqa(
        self, increment_des: bool = False
    ) -> Tuple[dict, dict, dict, bool]:
        """Generate an observation, question, and answer.

        Args
        ----
        increment_des: whether or not to take a step in the DES.

        Returns
        -------
        observation = {
            "human": <human>,
            "object": <obj>,
            "object_location": <obj_loc>,
        }
        question = {"human": <human>, "object": <obj>}
        answer = <obj_loc>
        is_last: True, if its the last observation in the queue, othewise False

        """
        first_human = self.human_sequence.pop(0)
        human_q = self.question_sequence.pop(0)

        is_last_o = len(self.human_sequence) == 0
        is_last_q = len(self.question_sequence) == 0

        assert is_last_o == is_last_q
        is_last = is_last_o

        if increment_des:
            self.des.step()

        first_object = self.des.state[first_human]["first_object"]
        relation = self.des.state[first_human]["relation"]
        second_human = self.des.state[first_human]["second_human"]
        second_object = self.des.state[first_human]["second_object"]
 
        observation = deepcopy(
            {
                "first_human": first_human,
                "first_object": first_object,
                "relation": relation,
                "second_human": second_human,
                "second_object": second_object,
                "current_time": self.des.current_time,
            }
        )

        if human_q is not None:
            # human_q is already a quintuple

            question = deepcopy(human_q)
            # NOTE: This is a dummy variable
            answer = 1 # deepcopy(obj_loc_q)  
        else:
            question = None
            answer = None

        return observation, question, answer, is_last

    def reset(self) -> dict:
        """Reset the environment.


        Returns
        -------
        state

        """
        self.des._initialize()
        self.generate_sequences()
        self.init_memory_systems()
        info = {}
        self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
            increment_des=False
        )


        encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
        encode_observation(self.ground_truth_memory_systems, "argmax", self.obs)

        state = deepcopy(self.extract_memory_entires(self.memory_systems))
        
        return state, info

    # TODO: Change this to allow training of both the memory management AND the memory filter
    # When you take a step, you idealy want to take in the action for memory management
    # That action may have already accounted for the filtered memory 
    # Also, pass in the filter returned by our policy. If it is passed in, then apply that filter to the memory to answer the question. Otherwise, use the entire memory 
    # However, allow the input of an answer to the question as well. If it is passed in, then don't use Steven's code but rather take a reward if the answer is correct. (This is for test time after supervised training)

    # Should return the correct filter using Steven's code AND the correct label answer to the question
    # Input: Dicionary containing 1. memory management action  2. filter bitmap (if None, then don't apply when answering question)  3. answer (if None, then either use Steven's code with or without filter)
    # Return: New observation, reward, done, truncated, info 
        # Info should contain: true filter map, true label answer

    # Training paradigm (easiest): Train memory management with manual QA (Steven's code). Then, train filter+classifier on collected data through supervised learning 
    # Training paradigm (medium): Train memory management in parallel/alternation with the filter output. However, binary classification is still handled by Steven's alg but on the filtered memories 
    # Training paradigm (hard): Train both the memory management AND the filter+classification in parallel. Only using Steven's code as our labels. Get reward for answers from trained classifier
    def step(self, actions:dict) -> Tuple[Tuple, int, bool, bool, dict]:  # last two are filter and correct answer (yes/no, 1/0)
        """An agent takes an action.

        Args
        ----
        action: This depends on the state

        Returns
        -------
        state, reward, done, truncated, info

        """
        info = {}
        truncated = False

        memory_action = actions["memory_management_action"]
        answer_action = actions["answer_action"]

        # memory_action will never be None. Assume that the corresponding policy is always working
        if memory_action == 0:
            manage_memory(self.memory_systems, "episodic")
        elif memory_action == 1:
            manage_memory(self.memory_systems, "semantic")
        elif memory_action == 2:
            manage_memory(self.memory_systems, "forget")
        else:
            raise ValueError
        
        # Insert the memory into the episodic for ground truth 
        manage_memory(self.ground_truth_memory_systems, "episodic")
        correct_answer = None
        if (self.question is None) and (self.answer is None):
            reward = 0
        else:
            if answer_action: # If not None, then an answer was passed into the method. Check that it is correct
                assert answer_action == 1 or answer_action == 0 or answer_action == 2
                pred = answer_action
            # else:  # Otherwise use the prediction by manually using the memory
                # if filter_action: # If there is a filter that was provided
                #     pred, correct_filter = answer_question(self.memory_systems, self.policies["question_answer"], self.question, filter_action)
            else:  # Then use the filter to answer the question manually
                pred = self.answer_generator.get_ans(self.question, self.memory_systems)
            
            # TODO: Use a different answer_generator for ground truth
            print("finding correct answer")
            correct_answer = self.answer_generator.get_ans(self.question, self.ground_truth_memory_systems)
            print("Correct answer: ", correct_answer)

            if pred != 2 and pred == correct_answer:
                reward = self.CORRECT
            else:
                reward = self.WRONG


        self.obs, self.question, self.answer, self.is_last = self.generate_oqa(
            increment_des=True
        )
        encode_observation(self.memory_systems, self.policies["encoding"], self.obs)

        # NOTE: Put the memory into the ground_truth memory system as well 
        encode_observation(self.ground_truth_memory_systems, "argmax", self.obs)


        state = deepcopy(self.extract_memory_entires(self.memory_systems))

        if self.is_last:
            done = True
        else:
            done = False

        info["correct_answer"] = correct_answer
        info["next_question"] = deepcopy(self.question)

        return state, reward, done, truncated, info

    def render(self, mode="console") -> None:
        if mode != "console":
            raise NotImplementedError()
        else:
            pass