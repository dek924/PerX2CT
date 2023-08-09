import torch
import torch.nn as nn

from x2ct_nerf.modules.losses.lpips import LPIPS
from utils.metrics import Peak_Signal_to_Noise_Rate_total, Structural_Similarity_slice, mse2psnr, img2mse

class EvaluationLoss(nn.Module):
    def __init__(self, perceptual_weight=1.0, ct_min_max=[0, 2500]):
        super().__init__()
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.ct_min_max = ct_min_max
        print(f"EvaluationLoss.")

    def forward(self, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
        nll_loss = rec_loss + self.perceptual_weight * p_loss
        nll_loss = torch.mean(nll_loss) ## for monitoring
        psnr = mse2psnr(img2mse(reconstructions.contiguous(), inputs.contiguous()))
        psnr_unnorm = \
        Peak_Signal_to_Noise_Rate_total(inputs.contiguous(), reconstructions.contiguous(), size_average=False,
                                        PIXEL_MIN=self.ct_min_max[0], PIXEL_MAX=self.ct_min_max[1])[-1]

        try:
            ssim = Structural_Similarity_slice(inputs.contiguous().detach().cpu().numpy(),
                                               reconstructions.contiguous().detach().cpu().numpy(), PIXEL_MAX=1.0)
        except:
            print(inputs.shape)
            print(reconstructions.shape)
            ssim = None

        log = {f"{split}/nll_loss": nll_loss.detach().mean(),
               f"{split}/rec_loss": rec_loss.detach().mean(),
               f"{split}/p_loss": p_loss.detach().mean(),
               f"{split}/psnr": psnr.detach().mean(),
               f"{split}/psnr_unnorm": psnr_unnorm.mean(),
               }
        if ssim is not None:
            log[f"{split}/ssim"]=ssim
        return log
