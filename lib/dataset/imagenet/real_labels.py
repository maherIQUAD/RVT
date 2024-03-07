import os
import json
import numpy as np

# Class for evaluating ImageNet predictions against "real" labels
class RealLabelsImagenet:

    # Initializes the evaluator
    def __init__(self, filenames, real_json='real.json', topk=(1, 5)):
        # Loads the real labels from a JSON file
        with open(real_json) as real_labels:
            real_labels = json.load(real_labels)
            # Converts the labels to a dictionary keyed by filename
            real_labels = {
                f'ILSVRC2012_val_{i + 1:08d}.JPEG': labels
                for i, labels in enumerate(real_labels)
            }
        self.real_labels = real_labels
        self.filenames = filenames
        # Ensures the number of filenames matches the number of real labels
        assert len(self.filenames) == len(self.real_labels)
        # Sets the top-k accuracy values to consider
        self.topk = topk
        # Initializes a dictionary to keep track of correctness for each k
        self.is_correct = {k: [] for k in topk}
        # Initializes the index for sample processing
        self.sample_idx = 0

    # Adds results from a model output
    def add_result(self, output):
        maxk = max(self.topk)
        _, pred_batch = output.topk(maxk, 1, True, True)
        pred_batch = pred_batch.cpu().numpy()
        for pred in pred_batch:
            # Retrieves the filename for the current sample
            filename = self.filenames[self.sample_idx]
            filename = os.path.basename(filename)
            # If the filename has real labels, check if predictions are correct
            if self.real_labels[filename]:
                for k in self.topk:
                    # Checks if the top-k predictions are in the real labels
                    self.is_correct[k].append(
                        any([p in self.real_labels[filename] for p in pred[:k]]))
            self.sample_idx += 1

    # Computes the accuracy for a given k or for all k values
    def get_accuracy(self, k=None):
        if k is None:
            # Returns a dictionary of accuracies for each k
            return {k: float(np.mean(self.is_correct[k])) for k in self.topk}
        else:
            # Returns the accuracy for the specified k
            return float(np.mean(self.is_correct[k])) * 100
