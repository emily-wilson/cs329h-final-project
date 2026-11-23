from dataclasses import dataclass
import numpy as np

## TODO: is there any way to have like a stabe chosen/rejected id? that way we don't have to diff the audio vectors
@dataclass
class Comparison:
    prompt: str
    chosen: np.ndarray;
    rejected: np.ndarray;
    user_id: str;

## Returns mean accuracy of comparisons
## predicted, ground truth -> length n, chosen, rejected -> (1, m) vector
def accuracy(predicted: list[Comparison], ground_truth: list[Comparison]):
    ## expected diff shape -> n x m
    diff = np.stack([p.chosen for p in predicted]) - np.stack([g.chosen for g in ground_truth])
    ## equal comparisons will be zero so we can just sum along axis 1
    diff = np.sum(diff, axis=1).flatten()
    ## expected diff shape now -> n x 1
    correct = np.equals(np.zeros((len(predicted))), diff)

    return correct / len(predicted)