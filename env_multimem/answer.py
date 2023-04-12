import json

# from .memory import *

json_file = '../data/env_multimem.json'
f = open(json_file)
data = json.load(f)

all_info = data["component_list"]
people = all_info["people"]
objects = all_info["objects"]
small_locations = all_info["small_locations"]
big_locations = all_info["large_locations"]
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

    def get_human(self):
        """Get the human of the object"""
        return self.human

    def get_big_loc(self):
        """Get the name of the object"""
        return self.name  

    def get_small_loc(self):
        """Get the small location of the object"""
        return self.small_loc

    def get_big_loc(self):
        """Get the big location of the object"""
        return self.big_loc   


class Answer:
    def __init__(self) -> None:
        self.all_objects = {}
        self.big_to_small = {}
        for big_loc in big_locations:
            self.big_to_small[big_loc] = set(())

        self.small_to_big = {}
        self.small_to_obj = {}
        for small_loc in small_locations:
            self.small_to_big[small_loc] = placeholder
            self.small_to_obj[small_loc] = set(())

    def locate_objects(self, memory):
        """REMINDER: Add get_memory function to memory.py"""
        self.short_episodic = memory["short"].get_memory() + memory["episodic"].get_memory()
        self.short_episodic.sort(key=lambda x: x["timestamp"])
        self.semantic = memory["semantic"].get_memory()

        self.semantic.sort(key=lambda x: x["num_generalized"], reverse = True)
        
        # print("start print memory+++++++++++")
        # print("short-episodic:")
        # print(self.short_episodic[:2])
        # print("semantic:")
        # print(self.semantic[:2])
        # print("end print memory+++++++++++++")
        
        # Locate each object through short + episodic
        # assume key for each memory slot: first_human, first_object, relation, second_human, second_object, timestamp
        # print("length", len(self.short_episodic))
        for info in self.short_episodic:
            # first item is object
            # print("+++", info)
            if info["first_object"] in objects:
                full_name = info["first_human"]+info["first_object"]
                if full_name in self.all_objects:
                    # already created, then update
                    obj1 = self.all_objects[full_name]
                    obj1_small_loc = obj1.get_small_loc()
                    if info["relation"] == "AtLocation":
                        if info["second_object"] in small_locations:
                            # object at small location
                            if obj1_small_loc != placeholder:
                                self.small_to_obj[obj1_small_loc].remove(full_name)
                            self.all_objects[full_name].update_small_loc(info["second_object"])
                            self.all_objects[full_name].update_big_loc(self.small_to_big[info["second_object"]])
                            self.small_to_obj[info["second_object"]].add(full_name)    
                        elif info["second_object"] in big_locations:
                            # object at big location
                            if obj1_small_loc != placeholder:
                                self.small_to_obj[obj1_small_loc].remove(full_name)
                            self.all_objects[full_name].update_small_loc(placeholder)
                            self.all_objects[full_name].update_big_loc(info["second_object"])
                    elif info["relation"] == "NextTo" :
                        # object next to object
                        if obj1_small_loc != placeholder:
                            self.small_to_obj[obj1_small_loc].remove(full_name)
                        obj2 = self.all_objects[info["second_human"]+info["second_object"]]
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
                        if info["second_object"] in small_locations:
                            # object at small location
                            self.all_objects[full_name] = Object({"human": info["first_human"],"object":info["first_object"],"small_location":info["second_object"], "big_location":placeholder})
                            self.small_to_obj[info["second_object"]].add(full_name)    
                        elif info["second_object"] in big_locations:
                            # object at big location
                            self.all_objects[full_name] = Object({"human": info["first_human"],"object":info["first_object"],"small_location":placeholder, "big_location":info["second_object"]})
                    elif info["relation"] == "NextTo" :
                        # object next to object
                        obj2 = self.all_objects[info["second_human"]+info["second_object"]]
                        obj2_small_loc = obj2.get_small_loc()
                        obj2_big_loc = obj2.get_big_loc()
                        if obj2_small_loc != placeholder:
                            self.small_to_obj[obj2_small_loc].add(full_name)
                        self.all_objects[full_name] = Object({"human": info["first_human"],"object":info["first_object"],"small_location":obj2_small_loc, "big_location":obj2_big_loc}) 
                    else:
                        # Error handle
                        pass      

            elif info["first_object"] in small_locations:
                # first item is small location
                if info["relation"] == "AtLocation":
                    if info["second_object"] in big_locations:
                        # small location at big location
                        if self.small_to_big[info["first_object"]] != placeholder:
                            self.big_to_small[self.small_to_big[info["first_object"]]].remove(info["first_object"])
                        self.small_to_big[info["first_object"]] = info["second_object"]
                        self.big_to_small[info["second_object"]].add(info["first_object"])
                        for obj_name in self.small_to_obj[info["first_object"]]:
                            self.all_objects[obj_name].update_big_loc(info["second_object"])
                    else:
                        # Error handle
                        pass
                elif info["relation"] == "NextTo":
                    if info["second_object"] in small_locations:
                        # small location next to small location
                        small_loc1_big = self.small_to_big[info["first_object"]]
                        small_loc2_big = self.small_to_big[info["second_object"]]
                        if small_loc1_big != placeholder:
                            self.big_to_small[small_loc1_big].remove(info["first_object"])
                        self.small_to_big[info["first_object"]] = small_loc2_big
                        if small_loc2_big != placeholder:
                            self.big_to_small[small_loc2_big].add(info["first_object"])
                        for obj_name in self.small_to_obj[info["first_object"]]:
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

        # build objects that only exists in semantic memory
        for info in self.semantic:
            if info["second_object"] in objects:
                obj2_name = info["second_human"] + info["second_object"]
                if obj2_name not in self.all_objects:
                    self.all_objects[obj2_name] = Object({"human": info["second_human"],"object":info["second_object"],"small_location":placeholder, "big_location":placeholder}) 
            if info["first_object"] in objects:
                obj1_name = info["first_human"] + info["first_object"]
                if obj1_name not in self.all_objects:
                    self.all_objects[obj1_name] = Object({"human": info["first_human"],"object":info["first_object"],"small_location":placeholder, "big_location":placeholder})    

        # Use semantic to fill blanks
        # assume key first_human, first_object, relation, second_human, second_object, strength
        # valid to first fill in small_loc?
        for obj_name in self.all_objects:
            small_loc = self.all_objects[obj_name].get_small_loc()
            big_loc = self.all_objects[obj_name].get_big_loc()
            if small_loc != placeholder and big_loc != placeholder:
                pass
            if small_loc == placeholder:
                for info in self.semantic:
                    if obj_name == info["first_human"]+ info["first_object"]:
                        if info["relation"] == "AtLocation":
                            if info["second_object"] in small_locations:
                                # semantic object at small location
                                self.all_objects[obj_name].update_small_loc(info["second_object"])
                                # new added branch
                                possible_big_loc = self.small_to_big[info["second_object"]]
                                if big_loc == placeholder and possible_big_loc != placeholder:
                                    self.all_objects[obj_name].update_big_loc(possible_big_loc)    
                                break
                            elif info["second_object"] in big_locations and big_loc == placeholder:
                                # semantic object at big location
                                self.all_objects[obj_name].update_big_loc(info["second_object"])
                        elif info["relation"] == "NextTo":
                            # semantic first_object next to second_object (we want is obj1)
                            obj2 = self.all_objects[info["second_human"]+info["second_object"]]
                            obj2_small_loc = obj2.get_small_loc()
                            obj2_big_loc = obj2.get_big_loc()
                            if obj2_small_loc != placeholder:
                                self.all_objects[obj_name].update_small_loc(obj2_small_loc)
                                if obj2_big_loc != placeholder and big_loc == placeholder:
                                    self.all_objects[obj_name].update_big_loc(obj2_big_loc)
                                break        
                    elif obj_name == info["second_human"]+ info["second_object"] and info["relation"] == "NextTo":
                        # semantic first_object next to second_object (we want is obj2)
                        obj1 = self.all_objects[info["first_human"]+info["first_object"]]
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
                    if obj_name == info["first_human"]+ info["first_object"]:
                        if info["relation"] == "AtLocation":
                            if info["second_object"] in big_locations:
                                # semantic object at big location
                                self.all_objects[obj_name].update_big_loc(info["second_object"])
                                break
                        elif info["relation"] == "NextTo":
                            # semantic first_object next to second_object (we want is obj1)
                            obj2 = self.all_objects[info["second_human"]+info["second_object"]]
                            obj2_small_loc = obj2.get_small_loc()
                            obj2_big_loc = obj2.get_big_loc()
                            if obj2_small_loc == small_loc:
                                self.all_objects[obj_name].update_big_loc(obj2_big_loc)
                                break        
                    elif obj_name == info["second_human"]+ info["second_object"] and info["relation"] == "NextTo":
                        # semantic first_object next to second_object (we want is obj2)
                        obj1 = self.all_objects[info["first_human"]+info["first_object"]]
                        obj1_small_loc = obj1.get_small_loc()
                        obj1_big_loc = obj1.get_big_loc()
                        if obj1_small_loc == small_loc:
                            self.all_objects[obj_name].update_big_loc(obj1_big_loc)
                            break     
                    elif info["first_object"] == self.all_objects[obj_name].get_small_loc():
                        if info["relation"] == "AtLocation":
                            if info["second_object"] in big_locations:
                                # semantic small location at big location
                                self.all_objects[obj_name].update_big_loc(info["second_object"])
                                break
                        elif info["relation"] == "NextTo":
                            if info["second_object"] in small_locations:
                                # semantic small location next to small location
                                if self.small_to_big[info["second_object"]] != placeholder:
                                    self.all_objects[obj_name].update_big_loc(self.small_to_big[info["second_object"]])
                                    break                       

        

    def print_obj_state(self):
        for item in self.all_objects:
            print("{human:", item.human, ", name:", item.name, ", small_loc:", item.small_loc, ", big_loc:", item.big_loc, "}")


    def get_ans(self, question, memory):
        """Assume the input as (human, object, at/nextto, human, object) and is valid"""
        """Here we do not consider nextto relation between different level"""
        """Not sure answer possible?"""
        self.locate_objects(memory)
        # print("printing objects")
        # for _, item in self.all_objects.items():
        #     print("{human:", item.get_human(), ", name:", item.get_name(), ", small_loc:", item.get_small_loc(), ", big_loc:", item.get_big_loc(), "}")
        # print("finish printing state")
        # print(question)
        if question[1] in objects:
            if question[2] == "AtLocation":
                # question object at small/big location
                obj = question[0] + question[1]
                # return question[4] == self.all_objects[obj].small_loc or question[4] == self.all_objects[obj].big_loc
                # question object at big location exists?
                return question[4] == self.all_objects[obj].small_loc
            elif question[2] == "NextTo":
                # question object next to object
                obj1 = question[0] + question[1]
                obj2 = question[3] + question[4]
                # return self.all_objects[obj1].big_loc == self.all_objects[obj2].big_loc and self.all_objects[obj1].small_loc == self.all_objects[obj2].small_loc
                return self.all_objects[obj1].small_loc == self.all_objects[obj2].small_loc
            else:
                # Error handle
                pass     
        elif question[1] in small_locations:
            if question[2] == "AtLocation":
                # question small location at big location
                return question[1] in self.big_to_small[question[4]]
            elif question[2] == "NextTo":
                #question small location next to small location
                for _,small_locs in self.big_to_small.items():
                    if question[1] in small_locs and question[4] in small_locs:
                        return True
                return False       
            else:
                # Error handle
                pass     
        else:
            # Error handle
            pass 


