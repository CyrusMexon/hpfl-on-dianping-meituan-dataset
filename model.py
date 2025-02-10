import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n):
        super(Net, self).__init__()
        self.knowledge_dim = knowledge_n

        # Original components
        self.student_emb = nn.Embedding(student_n, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(exer_n, self.knowledge_dim)

        # Enhanced for regression
        self.fc = nn.Sequential(
            nn.Linear(self.knowledge_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, stu_id, exer_id, kn_emb):
        # Original knowledge interaction
        stu_emb = self.student_emb(stu_id)
        ex_emb = self.k_difficulty(exer_id)

        # Enhanced feature processing
        combined = torch.cat(
            [
                stu_emb * ex_emb,
                kn_emb.mean(dim=1, keepdim=True).expand(-1, self.knowledge_dim),
                stu_emb,
            ],
            dim=1,
        )

        # Regression output (0-50 scale)
        output = torch.sigmoid(self.fc(combined))
        return output.squeeze()
