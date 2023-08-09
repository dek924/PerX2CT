import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from x2ct_nerf.modules.losses.lpips import LPIPS
from x2ct_nerf.modules.discriminator.model import ProgressiveDiscriminator

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

def softplus_d_loss(opt_idx, logits_real=None, logits_fake=None):
    if opt_idx == 0:    ## generator
        d_loss = F.softplus(-logits_fake).mean()
    else:   ## discriminator
        d_loss = F.softplus(logits_fake).mean() + F.softplus(-logits_real).mean()

    return d_loss


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, fade_steps=10000,
                 disc_factor=1.0, nll_factor=1.0,
                 perceptual_weight=1.0, lambda_d_r1=10,
                 disc_loss="softplus", recon_loss="L1"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "softplus"]
        assert recon_loss in ["L1", "L2"]
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.fade_steps = fade_steps
        self.lambda_d_r1 = lambda_d_r1
        self.nll_factor = nll_factor

        self.discriminator = ProgressiveDiscriminator()
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "softplus":
            self.disc_loss = softplus_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"LPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.recon_loss_type = recon_loss

    def forward(self, codebook_loss=None, inputs=None, reconstructions=None, optimizer_idx=None,
                global_step=None, last_layer=None, cond=None, split="train"):

        assert inputs is not None and reconstructions is not None and optimizer_idx is not None and global_step is not None
        diff = inputs.contiguous() - reconstructions.contiguous()
        if self.recon_loss_type == "L1":
            rec_loss = torch.abs(diff)
        elif self.recon_loss_type == "L2":
            rec_loss = (diff ** 2)

        if self.perceptual_weight > 0 or split != 'train':
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            #rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.zeros_like(rec_loss).to(diff.get_device())

        nll_loss = rec_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss)
        alpha = min(1, global_step / self.fade_steps)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous(), alpha)
            g_loss = self.disc_loss(optimizer_idx, None, logits_fake)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = self.nll_factor * nll_loss + disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            inputs = inputs.contiguous().detach()
            reconstructions = reconstructions.contiguous().detach()

            if self.training:
                inputs.requires_grad = True

            logits_real = self.discriminator(inputs, alpha)
            logits_fake = self.discriminator(reconstructions, alpha)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(optimizer_idx, logits_real, logits_fake)

            if self.training:
                grad_real = torch.autograd.grad(outputs=logits_real.sum(), inputs=inputs, create_graph=True)
                grad_real = [p for p in grad_real][0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5 * self.lambda_d_r1 * grad_penalty
            else:
                device = d_loss.get_device()
                grad_penalty = torch.tensor([0.0]).to(device)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/grad_penalty".format(split): grad_penalty.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss + grad_penalty, log
