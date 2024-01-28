import torch
import torch.nn.functional as F

class FGSMAttack:
    def __init__(self, model, epsilon=0.1):
        """
        Initialize the FGSM attack.

        :param model: The model to attack.
        :param epsilon: The maximum perturbation allowed.
        """
        self.model = model
        self.epsilon = epsilon

    def generate(self, x, y):
        """
        Generate an adversarial example using the FGSM attack.

        :param x: Input tensor (image).
        :param y: True label tensor.
        :return: Adversarial example tensor.
        """
        perturbed_x = x.clone().detach().to(x.device)
        perturbed_x.requires_grad = True

        output = self.model(perturbed_x)
        loss = F.nll_loss(output, y)

        self.model.zero_grad()

        loss.backward()

        with torch.no_grad():
            perturbed_x = perturbed_x + self.epsilon * perturbed_x.grad.sign()
            perturbed_x = torch.clamp(perturbed_x, 0, 1)

        return perturbed_x
