import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Utility function for linearly combining two tensors
def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

# Utility function for reducing loss to a scalar value
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

# Custom loss function class implementing label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    # Computes label-smoothed cross-entropy loss
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)

# Custom loss function class for training with soft targets
class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    # Computes cross-entropy loss using soft targets
    def forward(self, x, target):
        loss = th.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

# Function to build the appropriate criterion based on configuration
def build_criterion(config, train=True):
    # Criterion for training with mixup augmentation
    if config.AUG.MIXUP_PROB > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = SoftTargetCrossEntropy() if train else nn.CrossEntropyLoss()
    # Criterion with label smoothing
    elif config.LOSS.LABEL_SMOOTHING > 0.0 and config.LOSS.LOSS == 'softmax':
        criterion = LabelSmoothingCrossEntropy(config.LOSS.LABEL_SMOOTHING)
    # Standard CrossEntropyLoss for softmax
    elif config.LOSS.LOSS == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        # Raises error for unknown loss type
        raise ValueError('Unknown loss {}'.format(config.LOSS.LOSS))

    return criterion
