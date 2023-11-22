import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from importlib import import_module
from pytorch_lightning.utilities.distributed import rank_zero_only
import pdb

from main import instantiate_from_config
from utils.metrics import mse2psnr, img2mse

class BaseModel(pl.LightningModule):
    """
    without codebook, Use CT Only
    """
    def __init__(self,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 metadata={},
                 ):
        super().__init__()
        self.metadata = metadata
        self.image_key = image_key
        self.print_loss_per_step = self.metadata.get('print_loss_per_step', 50)
        net_module = self.get_net_module()
        net_params = self.metadata['net_params']
        self.net = net_module(**net_params)
        self.loss = instantiate_from_config(lossconfig)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))

    def get_net_module(self):
        net_module = self.metadata['net_module']
        net_module, net_class = net_module.rsplit(".", 1)
        net_module = getattr(import_module(net_module), net_class)
        return net_module

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input):
        output = self.net(input)
        output_dict = {
            'outputs': output,
        }
        return output_dict

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    @rank_zero_only
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step == 0:
            losses = ''
            for k, v in loss_dict.items():
                split, loss_name = k.split("/")
                losses = f"{losses}, {loss_name} : {float(v):.3f}"
            losses = f"[{split} Step{self.global_step}] : {losses[2:]}"
            print(f"{losses}")

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(x, xrec, optimizer_idx, self.global_step, split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec["outputs"], x)
            log_dict_ae['train/lr'] = self.learning_rate
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(x, xrec, optimizer_idx, self.global_step, split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec = self(x)
        aeloss, log_dict_ae = self.loss(x, xrec, 0, self.global_step, split="val")
        discloss, log_dict_disc = self.loss(x, xrec, 1, self.global_step, split="val")
        log_dict_ae['val/psnr'] = self.psnr(xrec["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.net.parameters()), lr=lr, betas=(0, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr*0.8, betas=(0, 0.9))
        return [opt_ae, opt_disc], []

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)["outputs"]
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

    def to_8b(self, x):
        return (255 * torch.clip(x, 0, 1)).type(torch.uint8)


class INRModel(BaseModel):
    """
    without encoder/decoder, Use CT, Xray
    """
    def __init__(self,
                 lossconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 metadata={},
                 ):
        super().__init__(lossconfig,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         metadata=metadata,
                         )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def get_input(self, batch):
        xs = {}
        for k, x in batch.items():
            if k in ['ctslice', 'PA', 'Lateral']:
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

    def forward(self, input, gt_ct=None, **kwargs):
        if 'p0' in kwargs.keys() and 'zoom_size' in kwargs.keys():
            output = self.net(input, p0=kwargs['p0'], zoom_size=kwargs['zoom_size'])
        else:
            output = self.net(input, gt_ct=gt_ct)
        if output["outputs"].shape[1] == 1:
            output["outputs"] = output["outputs"].repeat(1, 3, 1, 1)
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        #print("Training step")
        # pdb.set_trace()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict["outputs"], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                                split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #print("validation step")
        # pdb.set_trace()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            split="val")
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        x = batch[self.image_key]

        x = x.to(self.device)
        # encode
        x_rec = self(batch)["outputs"]

        for k, v in batch.items():
            if k in self.metadata['net_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

    @torch.no_grad()
    def log_images_from_gt(self, batch):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        x = batch[self.image_key]
        x = x.to(self.device)
        # encode
        x_rec = self(batch, x)

        for k, v in batch.items():
            if k in self.metadata['net_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log
