import torch
import torch.nn as nn
import torch.optim as optim

class CWAttack:
    def __init__(self, model, c=1e-3, kappa=0, max_iter=1000, learning_rate=0.01):
        """
        Initialize the C&W attack.

        :param model: The model to be attacked.
        :param c: Confidence parameter, higher value means more confidence in adversarial examples.
        :param kappa: Margin parameter, controls the difference between the target and other classes.
        :param max_iter: Maximum number of iterations for optimization.
        :param learning_rate: Learning rate for optimization.
        """
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def generate(self, x, y):
        """
        Generate adversarial example for the given input.

        :param x: Input tensor (image).
        :param y: True label of the input tensor.
        :return: Adversarial example.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Define the loss function
        def f(x_adv):
            outputs = self.model(x_adv)
            one_hot_labels = torch.eye(len(outputs[0]))[y].to(x.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_labels.bool())
            return torch.clamp(j - i, min=-self.kappa).mean()

        # Initialize perturbation
        w = torch.zeros_like(x, requires_grad=True)
        optimizer = optim.Adam([w], lr=self.learning_rate)

        for step in range(self.max_iter):
            a = 0.5 * (nn.Tanh()(w) + 1)
            loss1 = nn.MSELoss(reduction='sum')(a, x)
            loss2 = self.c * f(a)
            loss = loss1 + loss2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create adversarial example
        x_adv = 0.5 * (nn.Tanh()(w) + 1)

        return x_adv.detach()

# Usage
# model = ...  # Define your PyTorch model
# cw_attack = CWAttack(model)
# x_adv = cw_attack.generate(x, y)  # x is the input tensor, y is the true label
