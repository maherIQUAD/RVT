import torch
import torch.nn.functional as F


class PGDAttack:
    def __init__(self, model, epsilon=0.1, alpha=0.01, num_steps=40):
        """
        Initialize the PGD attack.
        
        :param model: The model to attack.
        :param epsilon: The maximum perturbation allowed.
        :param alpha: Step size per iteration.
        :param num_steps: Number of iterations for the attack.
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        
        

    def generate(self, x, y):
        """
        Generate an adversarial example using the PGD attack.

        :param x: Input tensor (image).
        :param y: True label tensor.
        :return: Adversarial example tensor.
        """
        perturbed_x = x.clone().detach().to(x.device)
        perturbed_x.requires_grad = True

        for step in range(self.num_steps):
            print('step: ', step)
            output = self.model(perturbed_x)
            loss = F.nll_loss(output, y)

            self.model.zero_grad()

            loss.backward()

            # Perform update outside of the no_grad() block
            perturbed_x = perturbed_x + self.alpha * perturbed_x.grad.sign()
            perturbed_x = torch.clamp(perturbed_x, 0, 1)

            with torch.no_grad():
                perturbation = torch.clamp(perturbed_x - x, -self.epsilon, self.epsilon)
                perturbed_x = x + perturbation

            # Ensure perturbed_x requires grad after updates
            perturbed_x = perturbed_x.detach().to(x.device)
            perturbed_x.requires_grad = True

        return perturbed_x

