import torch
import torchmetrics
from torchmetrics.utilities import dim_zero_cat

class EER(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("scores", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.scores.append(preds)
        self.labels.append(target)

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        self.update(preds, target)

    def compute(self):
        # Concatenate all scores and labels
        scores = dim_zero_cat(self.scores)
        labels = dim_zero_cat(self.labels)

        # Sort scores and calculate thresholds
        sorted_indices = scores.argsort()
        sorted_labels = labels[sorted_indices]

        # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
        far = (sorted_labels == 0).cumsum(0) / (sorted_labels == 0).sum()
        frr = 1 - (sorted_labels == 1).cumsum(0) / (sorted_labels == 1).sum()

        # Find the threshold where FAR and FRR are closest
        eer_index = (far - frr).abs().argmin()
        eer = (far[eer_index] + frr[eer_index]) / 2

        return eer