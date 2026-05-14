import torch
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # x: (input_size,) or (1, input_size)
        x = x.float().view(1, -1)
        return self.linear(x)  # logits: (1, num_classes)

class Readout:
    def __init__(self, input_size, num_classes, seed):
        self.input_size = input_size
        self.num_classes = num_classes  
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.manual_seed(self.seed)

        self._init_model()

    def _init_model(self):

        torch.manual_seed(self.seed)
        self.model = NN(self.input_size, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )


    def train_readout(self, training_pairs, n_epochs):
        self.model.train()
        pbar = tqdm(range(n_epochs), desc="Training readout")
        for epoch in pbar:
            total_loss = 0.0

            import random
            shuffled_pairs = training_pairs.copy()
            random.shuffle(shuffled_pairs)

            for s, l in shuffled_pairs:
                self.optimizer.zero_grad()
                logits = self.model(s)  # (1, C)
                target = torch.tensor([int(l)], dtype=torch.long)
                loss = self.criterion(logits, target)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            avg_loss = total_loss / len(training_pairs)
            self.scheduler.step(avg_loss)
            pbar.set_description_str(
                f"Epoch: {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}"
            )

    def test_readout(self, test_pairs):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for s, label in test_pairs:
                logits = self.model(s)              # (1, C)
                pred = logits.argmax(dim=1).item()  # int

                total += 1
                correct += int(pred == int(label))

        return 100.0 * correct / total if total else 0.0
