"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""
import random

from .memory import ShortMemory


def encode_observation(memory_systems: dict, policy: str, obs: dict) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "argmax" or "neural"
    obs: observation = {"human": <human>,
                        "object": <obj>,
                        "object_location": <obj_loc>}

    """
    if policy.lower() == "argmax":
        mem_short = ShortMemory.ob2short(obs)
    else:
        raise NotImplementedError

    memory_systems["short"].add(mem_short)


def manage_memory(memory_systems: dict, policy: str) -> None:
    """Non RL memory management policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic", "semantic", "forget", "random", or "neural"

    """
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "neural",
    ]
    if policy.lower() == "episodic":
        if memory_systems["episodic"].capacity != 0:
            if memory_systems["episodic"].is_full:
                memory_systems["episodic"].forget_oldest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems["episodic"].add(mem_epi)

    elif policy.lower() == "semantic":
        if memory_systems["semantic"].capacity != 0:
            if memory_systems["semantic"].is_full:
                memory_systems["semantic"].forget_weakest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems["semantic"].add(mem_sem)

    elif policy.lower() == "forget":
        pass

    elif policy.lower() == "random":
        action_number = random.choice([0, 1, 2])
        if action_number == 0:
            if memory_systems["episodic"].is_full:
                memory_systems["episodic"].forget_oldest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems["episodic"].add(mem_epi)

        elif action_number == 1:
            if memory_systems["semantic"].is_full:
                memory_systems["semantic"].forget_weakest()

            mem_short = memory_systems["short"].get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems["semantic"].add(mem_sem)

        else:
            pass

    elif policy.lower() == "neural":
        raise NotImplementedError

    else:
        raise ValueError

    memory_systems["short"].forget_oldest()


def answer_question(memory_systems: dict, policy: str, question: dict, memory_filter:dict | None) -> str:
    """Non RL question answering policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
            "random", or "neural",
    question: question = {"human": <human>, "object": <obj>}

    Returns
    -------
    pred: prediction

    """
    # TODO: Modify this to allow for accounting for binary filters
    assert policy.lower() in [
        "episodic_semantic",
        "semantic_episodic",
        "episodic",
        "semantic",
        "random",
        "neural",
    ]
    if memory_filter: # If memory filter is not None 
        episodic_filter = memory_filter["episodic"]
        semantic_filter = memory_filter["semantic"]
    else:
        episodic_filter = None 
        semantic_filter = None

    # TODO: The rest will be up to Steven here
    pred_epi, _ = memory_systems["episodic"].answer_latest(question, episodic_filter)
    pred_sem, _ = memory_systems["semantic"].answer_strongest(question, semantic_filter)

    if policy.lower() == "episodic_semantic":
        if pred_epi is None:
            pred = pred_sem
        else:
            pred = pred_epi
    elif policy.lower() == "semantic_episodic":
        if pred_sem is None:
            pred = pred_epi
        else:
            pred = pred_sem
    elif policy.lower() == "episodic":
        pred = pred_epi
    elif policy.lower() == "semantic":
        pred = pred_sem
    elif policy.lower() == "random":
        pred = random.choice([pred_epi, pred_sem])
    elif policy.lower() == "neural":
        raise NotImplementedError
    else:
        raise ValueError

    return pred