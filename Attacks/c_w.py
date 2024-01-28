import torch
import torch.optim as optim
import torch.nn.functional as F

class CWAttack:
    def __init__(self, model, c=1e-3, kappa=0, max_iter=1000, learning_rate=0.01):
        """
        Initialize the Carlini & Wagner (C&W) attack.

        :param model: The model to attack.
        :param c: Confidence parameter.
        :param kappa: Margin parameter for the attack.
        :param max_iter: Maximum number of iterations.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.model = model
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def generate(self, x, y):
        """
        Generate an adversarial example using the C&W attack.

        :param x: Input tensor (image).
        :param y: True label tensor.
        :return: Adversarial example tensor.
        """
        w = torch.zeros_like(x, requires_grad=True)

        optimizer = optim.Adam([w], lr=self.learning_rate)

        for step in range(self.max_iter):
            adv_x = w + x
            adv_x = torch.clamp(adv_x, 0, 1)

            outputs = self.model(adv_x)
            f_x_y = outputs[0, y]

            cost = torch.max(f_x_y - torch.max(outputs[0, 1-y:]), -self.kappa)
            loss = self.c * cost + torch.norm(w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adv_x = w + x
        adv_x = torch.clamp(adv_x, 0, 1)
        return adv_x.detach()
