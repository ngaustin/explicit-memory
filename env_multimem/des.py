"""Discrete Event Simlator (RoomDes) for the Room."""
import json
import os
from copy import deepcopy
from pprint import pprint


def read_json(fname: str) -> dict:
    """Read json.

    There is some path magic going on here. This is to account for both the production
    and development mode. Don't use this function for a general purpose.

    """
    fullpath = os.path.join(os.path.dirname(__file__), fname)

    with open(fullpath, "r") as stream:
        return json.load(stream)


class RoomDes:

    """RoomDes Class.

    This class is very simple at the moment. When it's initialized, it places N_{humans}
    in the room. They periodically move to other locations. They also periodically
    change the location of their objects they are holding. At the moment,
    everything is deterministic.

    """

    def __init__(self, des_size: str = "l", check_resources: bool = True) -> None:
        """Instantiate the class.

        Args
        ----
        des_size: configuartion for the RoomDes simulation. It should be either size or
            dict. size can be "xxs (extra extra small", "xs (extra small)", "s (small)",
            "m (medium)", or "l (large)".

            {"components": <COMPONENTS>, "resources": <RESOURCES>,
            "last_timestep": <LAST_TIMESTEP>,
            "semantic_knowledge": <SEMANTIC_KNOWLEDGE>, "complexity", <COMPLEXITY>}

            <COMPONENTS> should look like this:

            <RESOURCES> should look like this:

            {'desk': 2, 'A': 10000, 'lap': 10000}

            <LAST_TIMESTEP> is a number where the DES terminates.

            <SEMANTIC_KNOWLEDGE> is a dictionary of semantic knowledge.

            <COMPLEXITY> is defined as num_humans * num_total_objects
            * maximum_num_objects_per_human * maximum_num_locations_per_object

        check_resources: whether to check if the resources are depleted or not.

        """
        if isinstance(des_size, str):
            assert des_size.lower() in [
                "dev",
                "xxs",
                "xs",
                "s",
                "m",
                "l",
            ]
            # TODO: Create a json file that consists of humans, objects, small locations, and big locations
            self.config = read_json(f"./data/des-config-{des_size.lower()}-v1.json")
        else:
            self.config = des_size
        self.check_resources = check_resources
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the simulator."""
        self.components = deepcopy(self.config["components"])
        self.resources = deepcopy(self.config["resources"])
        self.semantic_knowledge = deepcopy(self.config["semantic_knowledge"])

        # TODO: Also initialize small/big locations
        
        self.first_humans = []
        self.first_objects = []
        self.relations = []
        self.second_humans = []
        self.second_object = []

        # TODO: Components will no longer be about just humans but also objects next to other objects, or at small locations etc. 
        for first_human, info in self.components.items():

            # First human and second human may be None, depending on whether the relation requires it...options listed below

            # 1. (Human, Object, NextTo, Human Object) 
            # 2. (Human, Object, AtLocation, None, SmallLocation)
            # 3. (None, SmallLocation, AtLocation, None, BigLocation)
            # 4. (None, SmallLocation, NextTo, None, SmallLocation) 


            # TODO: Each item in obj_locs will be: 
            # object, relation, object or something 
            self.first_humans.append(first_human)

            for first_object, relation, second_human, second_object in info:
                self.first_objects.append(first_object)
                self.relations.append(relation)
                self.second_humans.append(second_human)
                self.second_objects.append(second_object)

        self.first_humans = sorted(list(set(self.first_humans)))
        self.first_objects = sorted(list(set(self.first_objects)))
        self.relations = sorted(list(set(self.relations)))
        self.second_humans = sorted(list(set(self.second_humans)))
        self.object_locations = sorted(list(set(self.second_objects)))

        # TODO: Construct dictionaries to help answer questions 
        # TODO: This will the part for Steven's code..where we answer questions? 
        

        self.until = deepcopy(self.config["last_timestep"])

        self.states = []
        self.state = {}

        for human in self.components:
            self.state[human] = {}

            (
                self.state[human]["first_object"],
                self.state[human]["relation"],
                self.state[human]["second_human"],
                self.state[human]["second_object"]
            ) = self.components[human][0]

            self.state[human]["current_time"] = 0

        self.states.append(deepcopy(self.state))

        self.events = []
        self.current_time = 0

    def step(self) -> None:
        """Proceed time by one."""
        previous_state = deepcopy(self.state)
        previous_resources = deepcopy(self.resources)

        self.current_time += 1

        for human in self.state:
            object_location_idx = self.current_time % len(self.components[human])

            self.state[human]["current_time"] = self.current_time


            # TODO: Stochastically sample for new locations here but insert additional constraints
            # For example, a small location can only change big locations IF there are no objects on AtLocation to it 

            (
                self.state[human]["first_object"],
                self.state[human]["relation"],
                self.state[human]["second_human"],
                self.state[human]["second_object"]
            ) = self.components[human][object_location_idx]

        current_state = deepcopy(self.state)
        current_resources = deepcopy(self.resources)
        self.event = self.check_event(
            previous_state, previous_resources, current_state, current_resources
        )
        self.events.append(deepcopy(self.event))
        self.states.append(deepcopy(self.state))

    def check_event(
        self,
        previous_state: dict,
        previous_resources: dict,
        current_state: dict,
        current_resources: dict,
    ) -> dict:
        """Check if any events have occured between the two consecutive states.

        Args
        ----
        previous_state
        previous_resources
        current_state
        current_resources

        Returns
        -------
        event

        """
        assert len(previous_state) == len(current_state)
        assert len(previous_resources) == len(current_resources)

        state_changes = {}
        resource_changes = {}

        humans = list(previous_state)
        for human in humans:
            previous_object = previous_state[human]["object"]
            previous_object_location = previous_state[human]["object_location"]
            previous_time = previous_state[human]["current_time"]

            current_object = current_state[human]["object"]
            current_object_location = current_state[human]["object_location"]
            current_time = current_state[human]["current_time"]

            assert current_time == previous_time + 1

            state_changes[human] = {}

            if previous_object != current_object:
                state_changes[human]["object"] = {
                    "previous": previous_object,
                    "current": current_object,
                }
            if previous_object_location != current_object_location:
                state_changes[human]["object_location"] = {
                    "previous": previous_object_location,
                    "current": current_object_location,
                }

            if len(state_changes[human]) == 0:
                del state_changes[human]
            else:
                state_changes[human]["current_time"] = current_time

        for resource in previous_resources:
            previous_amount = previous_resources[resource]
            current_amount = current_resources[resource]

            if previous_amount != current_amount:
                resource_changes[resource] = current_amount - previous_amount

        return {"state_changes": state_changes, "resource_changes": resource_changes}

    def run(self, debug: bool = False) -> None:
        """Run until the RoomDes terminates."""
        while self.until > 0:
            self.step()
            if debug:
                pprint(self.event)

            self.until -= 1