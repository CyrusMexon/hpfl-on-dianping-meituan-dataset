import json
import torch
import numpy as np


class TrainDataLoader(object):
    """
    Data loader for training (modified for our structure)
    """

    def __init__(self, school, path):
        self.batch_size = 32
        self.ptr = 0
        self.data = []
        self.school = school
        data_file = f"{path}/train/{school}.json"
        # CHecking which path the JSON is being loaded from
        print("Loading JSON from:", data_file)

        config_file = "config.txt"

        with open(data_file, encoding="utf8") as i_f:
            self.data = json.load(i_f)

        config = np.loadtxt(config_file, delimiter=" ", dtype=int)
        index = np.where(config.T[0] == int(school))[0][0]
        self.student_n = config[index][1]
        self.exer_n = config[index][2]
        self.knowledge_n = config[index][3]
        self.knowledge_dim = self.knowledge_n

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []

        # Get current batch
        batch_data = self.data[self.ptr : self.ptr + self.batch_size]

        for log in batch_data:
            # Create knowledge embedding
            knowledge_emb = [0.0] * self.knowledge_dim
            for code in log["knowledge_code"]:
                if code <= self.knowledge_dim:  # Ensure code is within dimension
                    knowledge_emb[code - 1] = 1.0

            input_stu_ids.append(log["user_id"])
            input_exer_ids.append(log["exer_id"])
            input_knowedge_embs.append(knowledge_emb)
            ys.append(log["score"])

        self.ptr += self.batch_size
        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowedge_embs),
            torch.FloatTensor(ys),
        )

    def is_end(self):
        return self.ptr + self.batch_size > len(self.data)

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    """
    Data loader for validation/test (modified for flat structure)
    """

    def __init__(self, school, path):
        self.ptr = 0
        self.data = []
        self.school = school
        data_file = f"{path}/test/{school}.json"
        config_file = "config.txt"

        with open(data_file, encoding="utf8") as i_f:
            self.data = json.load(i_f)

        config = np.loadtxt(config_file, delimiter=" ", dtype=int)
        index = np.where(config.T[0] == int(school))[0][0]
        self.student_n = config[index][1]
        self.exer_n = config[index][2]
        self.knowledge_n = config[index][3]
        self.knowledge_dim = self.knowledge_n

    def next_batch(self):
        if self.is_end():
            return None, None, None, None, [], None

        # Get current batch (single user's interactions)
        batch_data = self.data[self.ptr : self.ptr + 1]
        self.ptr += 1

        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        know = []

        for log in batch_data:
            # Create knowledge embedding
            knowledge_emb = [0.0] * self.knowledge_dim
            current_know = []
            for code in log["knowledge_code"]:
                if code <= self.knowledge_dim:
                    knowledge_emb[code - 1] = 1.0
                    current_know.append(code)

            input_stu_ids.append(log["user_id"])
            input_exer_ids.append(log["exer_id"])
            input_knowledge_embs.append(knowledge_emb)
            ys.append(log["score"])
            know.append(current_know)

        return (
            torch.LongTensor(input_stu_ids),
            torch.LongTensor(input_exer_ids),
            torch.Tensor(input_knowledge_embs),
            torch.FloatTensor(ys),
            know,
            None,  # Teacher field not used in our data
        )

    def is_end(self):
        return self.ptr >= len(self.data)

    def reset(self):
        self.ptr = 0
