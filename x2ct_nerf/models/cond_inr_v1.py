import copy
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_only
import pdb

from main import instantiate_from_config
from utils.metrics import mse2psnr, img2mse
from x2ct_nerf.modules.losses.lpips import LPIPS
from x2ct_nerf.modules.utils import set_requires_grad

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Net2NetINRVQv1(pl.LightningModule):
    def __init__(self,
                 inr_config,
                 first_stage_config,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 metadata={}
                 ):
        super().__init__()
        self.metadata = metadata
        self.print_loss_per_step = metadata.get('print_loss_per_step', 50)
        self.first_stage_key = first_stage_key  ## ctslice
        self.first_stage_n_embed = first_stage_config['params']['n_embed']
        self.define_model(first_stage_config, inr_config, ckpt_path, ignore_keys)
        self.inr.quantize.beta = 0.
        self.inr.quantize.legacy = True

        self.loss = instantiate_from_config(lossconfig)
        self.cond_list = inr_config['params']['metadata']['encoder_params']['params']['cond_list']
        if monitor is not None:
            self.monitor = monitor

        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def define_model(self, first_stage_config, inr_config, ckpt_path, ignore_keys):
        self.inr = instantiate_from_config(config=inr_config)
        model = instantiate_from_config(first_stage_config)
        model.freeze()
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.inr.quantize = copy.deepcopy(model.quantize)
        self.inr.post_quant_conv = copy.deepcopy(model.post_quant_conv)
        self.inr.decoder = copy.deepcopy(model.decoder)

        del model.decoder
        del model.post_quant_conv
        del model.quantize

        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model


    def get_input(self, batch):
        xs = {'image_key': self.first_stage_key}
        for k, x in batch.items():
            if k in ['ctslice', 'PA', 'Lateral']:
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

    @rank_zero_only
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step == 0:
            losses = ''
            for k, v in loss_dict.items():
                split, loss_name = k.split("/")
                losses = f"{losses}, {loss_name} : {float(v):.3f}"
            losses = f"[{split} Step{self.global_step}] : {losses[2:]}"
            print(f"{losses}")

    def forward_first_stage(self, x):
        h = self.first_stage_model.encode_to_prequant(x)
        quant = self.inr.quantize(h)[0]
        dec = self.inr.decode(quant)
        output_dict = {
            'outputs': dec,
        }
        return output_dict

    def forward(self, input):
        z = self.inr.encode_to_prequant(input)
        zq, qloss, _ = self.inr.quantize(z['outputs'])
        xrec = self.inr.decode(zq)
        #xrec, qloss = self.inr(input)
        output_dict = {
            'outputs': xrec
        }
        return output_dict, qloss, z, zq

    def get_qloss_with_gt(self, batch, z):
        with torch.no_grad():
            x = batch[self.first_stage_key]
            h = self.first_stage_model.encode_to_prequant(x)
            zq_gt = self.inr.quantize(h)[0]
        qloss = torch.mean((zq_gt.detach() - z) ** 2)
        return qloss

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        xrec_dict, qloss, z, zq = self(batch)
        x = batch[self.first_stage_key]
        if self.metadata['qloss_with_gt']:
            qloss = self.get_qloss_with_gt(batch, z)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict['outputs'], x)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        xrec_dict, qloss, z, zq = self(batch)
        x = batch[self.first_stage_key]
        if self.metadata['qloss_with_gt']:
            qloss = self.get_qloss_with_gt(batch, z)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        log_dict_ae['val/psnr'] = self.psnr(xrec_dict['outputs'], x)
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(self.monitor, log_dict_ae[self.monitor],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.inr.encoder.parameters()) +
                                  list(self.inr.quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.inr.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        x = batch[self.first_stage_key]
        x = x.to(self.device)
        xrec = self.inr(batch)['outputs']
        x_rec_first = self.forward_first_stage(x)['outputs']

        for k, v in batch.items():
            if k in self.cond_list:
                log[k] = v

        log["inputs"] = x
        log["decoded"] = x_rec_first
        log["reconstructions"] = xrec
        return log
