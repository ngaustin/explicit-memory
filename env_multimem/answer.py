from .memory import *

people = ["Nature", "Michael", "Kai", "Cameron", "Joy", "Laura"],
objects = ["laptop", "bottle", "pencil", "cushion", "phone"],
small_locations = ["table", "couch", "bed", "chair", "stool"],
big_locations = ["Study", "Garage", "Living Room", "Kitchen"]
placeholder = "?"

class Object:
    def __init__(self, info) -> None:
        """Initialization of object"""
        # place holder '?'
        # MAYBE CHANGED LATER
        self.human = info["human"]
        self.name = info["object"]
        self.small_loc = info["small_location"]
        self.big_loc = info["big_location"]

    def update_small_loc(self, small_loc):
        """Update the small location of the object"""
        self.small_loc = small_loc

    def update_big_loc(self, big_loc):
        """Update the big location of the object"""
        self.big_loc = big_loc

    def get_small_loc(self):
        """Get the small location of the object"""
        return self.small_loc

    def get_big_loc(self):
        """Get the big location of the object"""
        return self.big_loc   


class Answer:
    def __init__(self, memory) -> None:
        self.memory = memory
        self.all_objects = {}
        self.big_to_small = {"Study":set(()), "Garage":set(()), "Living Room":set(()), "Kitchen":set(())}
        self.small_to_big = {"table":placeholder, "couch":placeholder, "bed":placeholder, "chair":placeholder, "stool":placeholder}
        self.small_to_obj = {"table":set(()), "couch":set(()), "bed":set(()), "chair":set(()), "stool":set(())}

    def locate_objects(self):
        """REMINDER: Add get_memory function to memory.py"""
        self.short_episodic = self.memory["short"].get_memory() + self.memory["episodic"].get_memory()
        self.short_episodic.sort(key=lambda x: x["timestamp"])
        self.semantic = self.memory["semantic"].get_memory()
        self.semantic.sort(key=lambda x: x["num_generalized"], reverse = True)

        # Locate each object through short + episodic
        # assume key for each memory slot: human1, object1, relation, human2, object2, timestamp
        for info in self.short_episodic:
            # first item is object
            if info["object1"] in objects:
                full_name = info["human1"]+info["object1"]
                if full_name in self.all_objects:
                    # already created, then update
                    obj1 = self.all_objects[full_name]
                    obj1_small_loc = obj1.get_small_loc()
                    if info["relation"] == "AtLocation":
                        if info["object2"] in small_locations:
                            # object at small location
                            if obj1_small_loc != placeholder:
                                self.small_to_obj[obj1_small_loc].remove(full_name)
                            self.all_objects[full_name].update_small_loc(info["object2"])
                            self.all_objects[full_name].update_big_loc(self.small_to_big[info["object2"]])
                            self.small_to_obj[info["object2"]].add(full_name)    
                        elif info["object2"] in big_locations:
                            # object at big location
                            if obj1_small_loc != placeholder:
                                self.small_to_obj[obj1_small_loc].remove(full_name)
                            self.all_objects[full_name].update_small_loc(placeholder)
                            self.all_objects[full_name].update_big_loc(info["object2"])
                    elif info["relation"] == "NextTo" :
                        # object next to object
                        if obj1_small_loc != placeholder:
                            self.small_to_obj[obj1_small_loc].remove(full_name)
                        obj2 = self.all_objects[info["human2"]+info["object2"]]
                        obj2_small_loc = obj2.get_small_loc()
                        obj2_big_loc = obj2.get_big_loc()
                        if obj2_small_loc != placeholder:
                            self.small_to_obj[obj2_small_loc].add(full_name)
                        self.all_objects[full_name].update_small_loc(obj2_small_loc)
                        self.all_objects[full_name].update_big_loc(obj2_big_loc)
                    else:
                        # Error handle
                        pass      
                else:
                    # object not created
                    if info["relation"] == "AtLocation":
                        if info["object2"] in small_locations:
                            # object at small location
                            self.all_objects[full_name] = Object({"human": info["human1"],"object":info["object1"],"small_location":info["object2"], "big_location":placeholder})
                            self.small_to_obj[info["object2"]].add(full_name)    
                        elif info["object2"] in big_locations:
                            # object at big location
                            self.all_objects[full_name] = Object({"human": info["human1"],"object":info["object1"],"small_location":placeholder, "big_location":info["object2"]})
                    elif info["relation"] == "NextTo" :
                        # object next to object
                        obj2 = self.all_objects[info["human2"]+info["object2"]]
                        obj2_small_loc = obj2.get_small_loc()
                        obj2_big_loc = obj2.get_big_loc()
                        if obj2_small_loc != placeholder:
                            self.small_to_obj[obj2_small_loc].add(full_name)
                        self.all_objects[full_name] = Object({"human": info["human1"],"object":info["object1"],"small_location":obj2_small_loc, "big_location":obj2_big_loc}) 
                    else:
                        # Error handle
                        pass      

            elif info["object1"] in small_locations:
                # first item is small location
                if info["relation"] == "AtLocation":
                    if info["object2"] in big_locations:
                        # small location at big location
                        if self.small_to_big[info["object1"]] != placeholder:
                            self.big_to_small[self.small_to_big[info["object1"]]].remove(info["object1"])
                        self.small_to_big[info["object1"]] = info["object2"]
                        self.big_to_small[info["object2"]].add(info["object1"])
                        for obj_name in self.small_to_obj[info["object1"]]:
                            self.all_objects[obj_name].update_big_loc(info["object2"])
                    else:
                        # Error handle
                        pass
                elif info["relation"] == "NextTo":
                    if info["object2"] in small_locations:
                        # small location next to small location
                        small_loc1_big = self.small_to_big[info["object1"]]
                        small_loc2_big = self.small_to_big[info["object2"]]
                        if small_loc1_big != placeholder:
                            self.big_to_small[small_loc1_big].remove(info["object1"])
                        self.small_to_big[info["object1"]] = small_loc2_big
                        if small_loc2_big != placeholder:
                            self.big_to_small[small_loc2_big].add(info["object1"])
                        for obj_name in self.small_to_obj[info["object1"]]:
                            self.all_objects[obj_name].update_big_loc(small_loc2_big)          
                    else:
                        # Error handle
                        pass
                else:
                    # Error handle
                    pass      

            else:
                # Error handle Any operation to big loc is NOT allowed  
                pass

        # Use semantic to fill blanks
        # assume key human1, object1, relation, human2, object2, strength
        # valid to first fill in small_loc?
        for obj_name in self.all_objects:
            small_loc = self.all_objects[obj_name].get_small_loc()
            big_loc = self.all_objects[obj_name].get_big_loc()
            if small_loc != placeholder and big_loc != placeholder:
                pass
            if small_loc == placeholder:
                for info in self.semantic:
                    if obj_name == info["human1"]+ info["object1"]:
                        if info["relation"] == "AtLocation":
                            if info["object2"] in small_locations:
                                # semantic object at small location
                                self.all_objects[obj_name].update_small_loc(info["object2"])
                                break
                            elif info["object2"] in big_locations and big_loc == placeholder:
                                # semantic object at big location
                                self.all_objects[obj_name].update_big_loc(info["object2"])
                        elif info["relation"] == "NextTo":
                            # semantic object1 next to object2 (we want is obj1)
                            obj2 = self.all_objects[info["human2"]+info["object2"]]
                            obj2_small_loc = obj2.get_small_loc()
                            obj2_big_loc = obj2.get_big_loc()
                            if obj2_small_loc != placeholder:
                                self.all_objects[obj_name].update_small_loc(obj2_small_loc)
                                if obj2_big_loc != placeholder and big_loc == placeholder:
                                    self.all_objects[obj_name].update_big_loc(obj2_big_loc)
                                break        
                    elif obj_name == info["human2"]+ info["object2"] and info["relation"] == "NextTo":
                        # semantic object1 next to object2 (we want is obj2)
                        obj1 = self.all_objects[info["human1"]+info["object1"]]
                        obj1_small_loc = obj1.get_small_loc()
                        obj1_big_loc = obj1.get_big_loc()
                        if obj1_small_loc != placeholder:
                            self.all_objects[obj_name].update_small_loc(obj1_small_loc)
                            if obj1_big_loc != placeholder and big_loc == placeholder:
                                self.all_objects[obj_name].update_big_loc(obj1_big_loc)
                            break
            
            # if after the filling big location still not sure
            if big_loc == placeholder:
                for info in self.semantic:
                    if obj_name == info["human1"]+ info["object1"]:
                        if info["relation"] == "AtLocation":
                            if info["object2"] in big_locations:
                                # semantic object at big location
                                self.all_objects[obj_name].update_big_loc(info["object2"])
                                break
                        elif info["relation"] == "NextTo":
                            # semantic object1 next to object2 (we want is obj1)
                            obj2 = self.all_objects[info["human2"]+info["object2"]]
                            obj2_small_loc = obj2.get_small_loc()
                            obj2_big_loc = obj2.get_big_loc()
                            if obj2_small_loc == small_loc:
                                self.all_objects[obj_name].update_big_loc(obj2_big_loc)
                                break        
                    elif obj_name == info["human2"]+ info["object2"] and info["relation"] == "NextTo":
                        # semantic object1 next to object2 (we want is obj2)
                        obj1 = self.all_objects[info["human1"]+info["object1"]]
                        obj1_small_loc = obj1.get_small_loc()
                        obj1_big_loc = obj1.get_big_loc()
                        if obj1_small_loc == small_loc:
                            self.all_objects[obj_name].update_big_loc(obj1_big_loc)
                            break               





    def get_ans(self, question):
        """Assume the input as (human, object, at/nextto, human, object) and is valid"""
        """Here we do not consider nextto relation between different level"""
        """Not sure answer possible?"""
        if question[1] in objects:
            if question[2] == "AtLocation":
                # question object at small/big location
                obj = question[0] + question[1]
                return question[4] == self.all_objects[obj].small_loc or question[4] == self.all_objects[obj].big_loc
            elif question[2] == "NextTo":
                # question object next to object
                obj1 = question[0] + question[1]
                obj2 = question[3] + question[4]
                return self.all_objects[obj1].big_loc == self.all_objects[obj2].big_loc and self.all_objects[obj1].small_loc == self.all_objects[obj2].small_loc
            else:
                # Error handle
                pass     
        elif question[1] in small_locations:
            if question[2] == "AtLocation":
                # question small location at big location
                return question[1] in self.big_to_small[question[4]]
            elif question[2] == "NextTo":
                #question small location next to small location
                for _,small_locs in self.big_to_small:
                    if question[1] in small_locs and question[4] in small_locs:
                        return True
                return False       
            else:
                # Error handle
                pass     
        else:
            # Error handle
            pass 


