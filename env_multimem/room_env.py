"""Room environment compatible with gym.

This env uses the RoomDes (room_env/envs/des.py), and Memory classes.
This is a more generalized version than RoomEnv0.
"""
import logging
import os
import random
from copy import deepcopy
from typing import Tuple
import math

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

        self.correct_answer_counter = {0: 0, 1:0, 2:0}
        self.num_correct = {0: 0, 1:0, 2:0}

        self.question_to_answer = {}  

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
                # print("obs num ", i, ": ", human, state[human])
                
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

                # print("generating sequences.... num candidate humans: {}, num candidate questions {}".format(len(human_candidates), len(possible_questions)))
                if len(possible_questions) == 0:
                    self.question_sequence.append([])
                    # print("question: ", [])
                else:
                    self.question_sequence.append(possible_questions.copy())
                    # print("possible questions: ", possible_questions.copy())

            assert len(possible_questions) == len(set(possible_questions))

            # self.des._initialize()

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

        effective_question_sequence = self.question_sequence[:-1]
        # for i, question in enumerate(self.question_sequence[:-1]):
        #     if random.random() < self.question_prob and question:  # make sure the question is not None
        #         effective_question_sequence.append(question)
        #     else:
        #         effective_question_sequence.append(None)
            
        # The last observation shouldn't have a question
        effective_question_sequence.append([])
        self.question_sequence = effective_question_sequence

        assert len(self.human_sequence) == len(self.question_sequence)

        self.num_questions = sum(
            [True for question in self.question_sequence if len(question) > 0]
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
        self, timestep=0
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
        human_q_options = self.question_sequence.pop(0)

        is_last_o = len(self.human_sequence) == 0
        is_last_q = len(self.question_sequence) == 0

        assert is_last_o == is_last_q
        is_last = is_last_o

        # if increment_des:
        #     self.des.step()
        state_human = self.des.get_step(timestep, first_human)

        first_object = state_human["first_object"]
        relation = state_human["relation"]
        second_human = state_human["second_human"]
        second_object = state_human["second_object"]
 
        observation = deepcopy(
            {
                "first_human": first_human,
                "first_object": first_object,
                "relation": relation,
                "second_human": second_human,
                "second_object": second_object,
                "current_time": timestep, # self.des.current_time,
            }
        )

        if len(human_q_options) != 0:
            # human_q_options is a list of possible questions for this state

            question = deepcopy(human_q_options)
            # NOTE: This is a dummy variable
            answer = None # deepcopy(obj_loc_q)  
        else:
            question = []
            answer = None

        return observation, question, answer, is_last
    
    def entropy(self, a_count):
        total_answers = sum(a_count)
        total = 0
        for i in range(3):
            if a_count[i] != 0:
                p = a_count[i] / total_answers
                total -= p * math.log(p)
        return total

    def reset(self) -> dict:
        """Reset the environment.


        Returns
        -------
        state

        """
        self.des._initialize()
        self.generate_sequences()
        self.init_memory_systems()
        self.timestep = 0
        info = {}
        # self.answer is a dummy question
        self.obs, question_options, self.answer, self.is_last = self.generate_oqa(
            timestep=0
        )

        # print(self.question_to_answer)
        entropies = []
        for q, a_count in self.question_to_answer.items():
            entropies.append(self.entropy(a_count))
        if len(entropies) > 0:
            print("Average entropy distribution of answers: ", sum(entropies) / len(entropies))

        if len(question_options) == 0:
            self.question = None
            self.answer = None
        else:
            assert NotImplementedError

        encode_observation(self.memory_systems, self.policies["encoding"], self.obs)
        encode_observation(self.ground_truth_memory_systems, "argmax", self.obs)

        state = deepcopy(self.extract_memory_entires(self.memory_systems))

        print("Correct answer counter after episode: ", self.correct_answer_counter, self.num_correct)
        
        self.correct_answer_counter = {0: 0, 1:0, 2:0}
        self.num_correct = {0: 0, 1:0, 2:0}
        
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

        # print("answer action in room_env: ", answer_action)

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
        if (self.question is None) and (self.answer is None):
            reward = 0
            correct_answer = 2  # This is a dummy variable
        else:
            if isinstance(answer_action, int): # If not None, then an answer was passed into the method. Check that it is correct
                assert answer_action == 1 or answer_action == 0
                pred = answer_action
            # else:  # Otherwise use the prediction by manually using the memory
                # if filter_action: # If there is a filter that was provided
                #     pred, correct_filter = answer_question(self.memory_systems, self.policies["question_answer"], self.question, filter_action)
            else:  # Then use the filter to answer the question manually
                # assert False
                # Populate the memory 
                self.answer_generator.locate_objects(self.memory_systems) 
                pred = self.answer_generator.get_ans(self.question) 
                assert pred != None and pred != 2

            # Initialize the memory system in the answerer
            self.answer_generator.locate_objects(self.ground_truth_memory_systems)
            correct_answer = self.answer_generator.get_ans(self.question)

            # print("Question distribution: ", self.question_to_answer.get(self.question, [0, 0, 0]))

            self.correct_answer_counter[correct_answer] += 1
            
            self.num_correct[correct_answer] += 1 if (pred == correct_answer) else 0
            assert correct_answer != None

            # print("Prediction: ", pred, "   Manual prediction: ", manual_pred, "    Correct answer:    ", correct_answer)
            if pred != 2 and pred == correct_answer:
                reward = self.CORRECT
            else:
                reward = self.WRONG
            
            # print("In Room_env   : ", answer_action, pred, correct_answer, reward)

        # self.answer is a dummy variable
        self.timestep += 1
        self.obs, question_options, self.answer, self.is_last = self.generate_oqa(
            timestep=self.timestep
        )
        
        encode_observation(self.memory_systems, self.policies["encoding"], self.obs)

        # NOTE: Put the memory into the ground_truth memory system as well 
        encode_observation(self.ground_truth_memory_systems, "argmax", self.obs)

        # Reinitialize the memory system in answerer after the new observation has been added
        self.answer_generator.locate_objects(self.ground_truth_memory_systems)

        # print("Ground truth memory for question: ")
        # print(self.ground_truth_memory_systems)
        random.shuffle(question_options)  # in-place shuffling
        if len(question_options) == 0:
            self.question = None
        else:
            intended_answer = random.choice(list(range(3)))
            found_question = False
            answers_found = []
            entropies = []
            max_entropy = math.log(2)
            self.question = None

            # Find question that not only maximizes the answer distribution's entropy but also equalizes the distribution of correct answers for questions
            for q in question_options: 
                expected_answer = self.answer_generator.get_ans(q)
                answers_found.append(expected_answer)
                
                curr_count_of_answers = self.question_to_answer.get(q, [0, 0, 0])
                curr_count_of_answers[expected_answer] += 1
                self.question_to_answer[q] = curr_count_of_answers

                e = self.entropy(curr_count_of_answers)
                entropies.append(e)

                if intended_answer == expected_answer:
                    found_question = True
                    p = e / max_entropy 
                    if random.random() < p: 
                        self.question = q 
                        break
            
            if not self.question:
                if found_question:
                    probs = [e if answers_found[i] == intended_answer else 0 for i, e in enumerate(entropies)]
                    if sum(probs) == 0:
                        self.question = random.choice([q for i, q in enumerate(question_options) if answers_found[i] == intended_answer])
                    else:
                        selection = random.choices(list(range(len(entropies))), weights=probs, k=1)[0]
                        self.question = question_options[selection]
                elif len(question_options) > 0:
                    # If didn't find it, then use a question that has had the lowest number of answers: 
                    filtered = [(answer, count) for answer, count in self.correct_answer_counter.items() if answer != intended_answer]
                    intended_answer = min(filtered, key=lambda x: x[1])[0]

                    filtered_entropies = []
                    for i, e in enumerate(entropies):
                        filtered_entropies.append(e if answers_found[i] == intended_answer else 0)
                        found_question = found_question or answers_found[i] == intended_answer
                    if found_question:
                        if sum(filtered_entropies) == 0:
                            self.question = random.choice([q for i, q in enumerate(question_options) if answers_found[i] == intended_answer])
                        else:
                            selection = random.choices(list(range(len(entropies))), weights=filtered_entropies, k=1)[0]
                            self.question = question_options[selection]
                    else:
                        self.question = random.choice(question_options)

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