"""Deep Q-network architecture. Currently only LSTM is implemented."""
import ast
from copy import deepcopy
from multiprocessing.sharedctypes import Value

import torch
from torch import nn

  
class LSTM(nn.Module):
    """A simple LSTM network."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        n_actions: int,
        embedding_dim: int,
        capacity: dict,
        entities: dict,
        include_human: str,
        batch_first: bool = True,
        memory_systems: list = ["episodic", "semantic", "short"],
        human_embedding_on_object_location: bool = False,
        accelerator: str = "cpu",
        use_filter: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the LSTM.

        Args
        ----
        hidden_size: hidden size of the LSTM
        num_layers: number of the LSTM layers
        n_actions: number of actions. This should be 3, at the moment.
        embedding_dim: entity embedding dimension (e.g., 32)
        capacity: the capacities of memory systems.
            e.g., {"episodic": 16, "semantic": 16, "short": 1}
        entities:
            e,g, {
            "humans": ["Foo", "Bar"],
            "objects": ["laptop", "phone"],
            "object_locations": ["desk", "lap"]}
        include_human:
            None: Don't include humans
            "sum": sum up the human embeddings with object / object_location embeddings.
            "cocnat": concatenate the human embeddings to object / object_location
                embeddings.
        batch_first: Should the batch dimension be the first or not.
        memory_systems: memory systems to be included as input
        human_embedding_on_object_location: whether to superposition the human embedding
            on the tail (object location entity).
        accelerator: "cpu", "gpu", or "auto"

        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.entities = entities
        self.include_human = include_human
        self.memory_systems = [ms.lower() for ms in set(memory_systems)]
        self.human_embedding_on_object_location = human_embedding_on_object_location
        self.use_filter = use_filter

        if accelerator == "gpu":
            self.device = "cuda"
        elif accelerator == "cpu":
            self.device = "cpu"
        else:
            raise ValueError

        self.create_embeddings()

        if "episodic" in self.memory_systems:
            self.lstm_e = nn.LSTM(
                self.input_size_e, hidden_size, num_layers, batch_first=batch_first
            )
            self.fc_e0 = nn.Linear(hidden_size, hidden_size)
            self.fc_e1 = nn.Linear(hidden_size, hidden_size)

            self.fc_e0_filter = nn.Linear(hidden_size, hidden_size)

        if "semantic" in self.memory_systems:
            self.lstm_s = nn.LSTM(
                self.input_size_s, hidden_size, num_layers, batch_first=batch_first
            )
            self.fc_s0 = nn.Linear(hidden_size, hidden_size)
            self.fc_s1 = nn.Linear(hidden_size, hidden_size)

            self.fc_s0_filter = nn.Linear(hidden_size, hidden_size)

        if "short" in self.memory_systems:
            self.lstm_o = nn.LSTM(
                self.input_size_o, hidden_size, num_layers, batch_first=batch_first
            )
            self.fc_o0 = nn.Linear(hidden_size, hidden_size)
            # self.fc_o0 = nn.Linear(self.input_size_o, hidden_size)
            self.fc_o1 = nn.Linear(hidden_size, hidden_size)

            self.fc_o0_filter = nn.Linear(hidden_size, hidden_size)

        self.fc_q0 = nn.Linear(self.embedding_dim * 5, hidden_size)  # Times 5 because question has 5 parts (Human1, Object1, Relation, Human2, Object2)
        self.fc_q1 = nn.Linear(hidden_size, hidden_size)

        self.fc_filter_q0 = nn.Linear(self.embedding_dim * 5, hidden_size)
        self.fc_filter_q1 = nn.Linear(hidden_size, hidden_size)

        self.lstm_filter_e = nn.LSTM(
                self.input_size_e, hidden_size, num_layers, batch_first=batch_first
            )

        self.lstm_filter_s = nn.LSTM(
                self.input_size_s, hidden_size, num_layers, batch_first=batch_first
            )
        
        self.lstm_filter_o = nn.LSTM(
                self.input_size_o, hidden_size, num_layers, batch_first=batch_first
            )

        self.fc_filter_all0 = nn.Linear(hidden_size * 4, max(hidden_size, sum(capacity.values())))
        self.fc_filter_all1 = nn.Linear(max(hidden_size, sum(capacity.values())), sum(capacity.values()))

        self.fc_final_question0 = nn.Linear(
            hidden_size * (len(self.memory_systems) + 1),   # Hidden size for the memory systems AND the question
            hidden_size * (len(self.memory_systems) + 1)
        )

        # Final output for the question is a multicass variable that is a yes/no/idk answer
        self.fc_final_question1 = nn.Linear(hidden_size * (len(self.memory_systems) + 1), 3)
        

        # Continue original code
        self.fc_final0 = nn.Linear(
            hidden_size * len(self.memory_systems),
            hidden_size * len(self.memory_systems),
        )
        self.fc_final1 = nn.Linear(hidden_size * len(self.memory_systems), n_actions)
        self.relu = nn.ReLU()

    def create_embeddings(self) -> None:
        """Create learnable embeddings."""
        self.word2idx = (
            ["<PAD>"]
            + self.entities["people"]
            + self.entities["objects"]
            + self.entities["small_locations"]
            + self.entities["big_locations"]
            + self.entities["relations"]
        )

        self.word2idx = {word: idx for idx, word in enumerate(self.word2idx)}
        self.embeddings = nn.Embedding(
            len(self.word2idx), self.embedding_dim, device=self.device, padding_idx=0
        )
        self.input_size_s = self.embedding_dim * 5
        self.input_size_o = self.embedding_dim * 5
        self.input_size_e = self.embedding_dim * 5

    
    def make_embedding_question(self, question: tuple) -> torch.Tensor:
        first_human, first_object, relation, second_human, second_object = question

        # Each item is already in word2idx because must be included in the environment to be a valid question 
        fh_idx = self.word2idx[first_human]
        fo_idx = self.word2idx[first_object]
        r_idx = self.word2idx[relation]
        sh_idx = self.word2idx[second_human]
        so_idx = self.word2idx[second_object]

        fh_embedding = self.embeddings(torch.tensor(fh_idx, device=self.device))
        fo_embedding = self.embeddings(torch.tensor(fo_idx, device=self.device))
        r_embedding = self.embeddings(torch.tensor(r_idx, device=self.device))
        sh_embedding = self.embeddings(torch.tensor(sh_idx, device=self.device))
        so_embedding = self.embeddings(torch.tensor(so_idx, device=self.device))

        final_embedding = torch.concat(
                [fh_embedding, fo_embedding, r_embedding, sh_embedding, so_embedding]
            )
        return final_embedding
    
    def create_batch_question(self, questions):
        batch = []
        for q in questions:
            question_tuple_from_string = ast.literal_eval(q) if type(q) == str else q
            batch.append(self.make_embedding_question(question_tuple_from_string))
        batch = torch.stack(batch)

        return batch

    def make_embedding(self, mem: dict, memory_type: str) -> torch.Tensor:
        """Create one embedding vector with summation and concatenation.

        Args
        ----
        mem: memory
            e.g, {"human": "Bob", "object": "laptop",
                  "object_location": "desk", "timestamp": 1}
        memory_type: "episodic", "semantic", or "short"

        Returns
        -------
        one embedding vector made from one memory element.

        """
        first_human_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["first_human"]], device=self.device)
        )
        first_object_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["first_object"]], device=self.device)
        )
        relation = self.embeddings(
            torch.tensor(self.word2idx[mem["relation"]], device=self.device)
        )
        second_human_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["second_human"]], device=self.device)
        )
        second_object_embedding = self.embeddings(
            torch.tensor(self.word2idx[mem["second_object"]], device=self.device)
        )

        final_embedding = torch.concat([first_human_embedding, first_object_embedding, relation, second_human_embedding, second_object_embedding])

        return final_embedding

    def create_batch(self, x: list, max_len: int, memory_type: str) -> torch.Tensor:
        """Create one batch from data.

        Args
        ----
        x: a batch of episodic, semantic, or short memories.
        max_len: maximum length (memory capacity)
        memory_type: "episodic", "semantic", or "short"

        Returns
        -------
        batch of embeddings.

        """
        batch = []

        for mems_str in x:
            entries = ast.literal_eval(mems_str)

            if memory_type == "semantic":
                mem_pad = {
                "first_human":"<PAD>", 
                "first_object":"<PAD>",
                "relation":"<PAD>",
                "second_human":"<PAD>",
                "second_object":"<PAD>",
                "num_generalized":"<PAD>"
            }
            elif memory_type in ["episodic", "short"]:
                 mem_pad = {
                "first_human":"<PAD>", 
                "first_object":"<PAD>",
                "relation":"<PAD>",
                "second_human":"<PAD>",
                "second_object":"<PAD>",
                "timestamp":"<PAD>"
            }
            else:
                raise ValueError

            for _ in range(max_len - len(entries)):
                # this is a dummy entry for padding.
                entries.append(mem_pad)
            mems = []
            for entry in entries:
                mem_emb = self.make_embedding(entry, memory_type)
                mems.append(mem_emb)
            mems = torch.stack(mems)
            batch.append(mems)
        batch = torch.stack(batch)

        return batch

    def forward(self, x: list) -> torch.Tensor:
        """Forward-pass.

        Args
        ----
        x[0]: episodic batch
            the length of this is batch size
        x[1]: semantic batch
            the length of this is batch size
        x[2]: short batch
            the length of this is batch size
        x[3]: question batch
            the length of this is batch size 
        """
        x_ = deepcopy(x)
        for i in range(len(x_)):
            if isinstance(x_[i], str):
                # bug fix. This happens when batch_size=1. I don't even know why
                # batch size 1 happens.
                x_[i] = [x_[i]]

        # TODO: Insert concat for the question/filter stuff as well?
        to_concat = []
        if "episodic" in self.memory_systems:
            batch_e = self.create_batch(
                x_[0], self.capacity["episodic"], memory_type="episodic"
            )
            lstm_out_e, _ = self.lstm_e(batch_e)
            fc_out_e = self.relu(
                self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
            )
            to_concat.append(fc_out_e)

        if "semantic" in self.memory_systems:
            batch_s = self.create_batch(
                x_[1], self.capacity["semantic"], memory_type="semantic"
            )
            lstm_out_s, _ = self.lstm_s(batch_s)
            fc_out_s = self.relu(
                self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
            )
            to_concat.append(fc_out_s)

        if "short" in self.memory_systems:
            batch_o = self.create_batch(
                x_[2], self.capacity["short"], memory_type="short"
            )
            lstm_out_o, _ = self.lstm_o(batch_o)
            fc_out_o = self.relu(
                self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
            )
            to_concat.append(fc_out_o)
        
        if len(x_) > 3: # Question was also passed in
            is_none = False
            # print("Episodic memory in model forward: ", x[0], "Semantic: ", x[1], "Short", x[2], "   Question in model forward: ", x[3])
            if len(x_[3]) == 1:  # singleton...no batch training but in steps 
                # print(x_[3])
                is_none = ast.literal_eval(x_[3][0])[0] == 1
            if len(x_[3]) > 0 and not is_none:  # Make sure not None
                # TODO: Create the filter here. FC batch_q, apply fc to the lstm_out_s, lstm_out_o, lstm_out_e, output filter, apply to create_batch output on the episodic, semantic, short memory, repass into the entire model
                batch_q = self.create_batch_question(x_[3])  
                memory_filter_out = None
                if self.use_filter:
                    to_concat = []

                    q_filter_out = self.fc_filter_q1(self.relu(self.fc_filter_q0(batch_q)))

                    e_filter_out = self.fc_e0_filter(self.lstm_filter_e(batch_e)[0][:, -1, :])
                    s_filter_out = self.fc_s0_filter(self.lstm_filter_s(batch_s)[0][:, -1, :])
                    o_filter_out = self.fc_o0_filter(self.lstm_filter_o(batch_o)[0][:, -1, :])

                    memory_filter_out = torch.concat([e_filter_out, s_filter_out, o_filter_out, q_filter_out], dim=1)
                    memory_filter_out = self.fc_filter_all1(self.relu(self.fc_filter_all0(memory_filter_out)))

                    # memory_filter_out = torch.relu(torch.sign((torch.nn.functional.sigmoid(memory_filter_out) - .5)))
                    memory_filter_out = torch.nn.functional.sigmoid(memory_filter_out)
                    
                    m = torch.distributions.Bernoulli(memory_filter_out)
                    memory_filter = m.sample()

                    # print("Sum of memory filter during forward: ", torch.sum(memory_filter))

                    e_filter = memory_filter[:, :self.capacity["episodic"]].unsqueeze(2)
                    s_filter = memory_filter[:, self.capacity["episodic"] : self.capacity["episodic"] + self.capacity["semantic"]].unsqueeze(2)
                    o_filter = memory_filter[:, self.capacity["episodic"] + self.capacity["semantic"]: ].unsqueeze(2)

                    filtered_batch_e = batch_e * e_filter 
                    filtered_batch_s = batch_s * s_filter 
                    filtered_batch_o = batch_o * o_filter 

                    lstm_out_e, _ = self.lstm_e(filtered_batch_e)
                    fc_out_e = self.relu(
                        self.fc_e1(self.relu(self.fc_e0(lstm_out_e[:, -1, :])))
                    )
                    to_concat.append(fc_out_e)

                    lstm_out_s, _ = self.lstm_s(filtered_batch_s)
                    fc_out_s = self.relu(
                        self.fc_s1(self.relu(self.fc_s0(lstm_out_s[:, -1, :])))
                    )
                    to_concat.append(fc_out_s)

                    lstm_out_o, _ = self.lstm_o(filtered_batch_o)
                    fc_out_o = self.relu(
                        self.fc_o1(self.relu(self.fc_o0(lstm_out_o[:, -1, :])))
                    )
                    to_concat.append(fc_out_o)
                
                res = self.fc_q1(self.relu(self.fc_q0(batch_q)))
                to_concat.append(res)
                fc_out_all = torch.concat(to_concat, dim=-1)
                fc_out = self.fc_final_question1(self.relu(self.fc_final_question0(fc_out_all)))
                # fc_out = torch.nn.functional.softmax(fc_out, dim=1)

                # TODO: Return the filter as well
                return fc_out, memory_filter_out
            else:
                return torch.tensor([[0, 0, 1]]), None 

        # dim=-1 is the feature dimension
        fc_out_all = torch.concat(to_concat, dim=-1)

        # fc_out has the dimension of (batch_size, 2)
        fc_out = self.fc_final1(self.relu(self.fc_final0(fc_out_all)))

        return fc_out, None
