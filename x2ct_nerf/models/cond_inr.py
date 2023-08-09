import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from pytorch_lightning.utilities.distributed import rank_zero_only
import pdb

from main import instantiate_from_config
from utils.metrics import mse2psnr, img2mse
from x2ct_nerf.modules.losses.lpips import LPIPS

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Net2NetINRGumbel(pl.LightningModule):
    def __init__(self,
                 inr_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 metadata={}
                 ):
        super().__init__()
        self.print_loss_per_step = metadata.get('print_loss_per_step', 50)
        self.first_stage_key = first_stage_key  ## ctslice
        self.first_stage_n_embed = first_stage_config['params']['n_embed']
        self.init_first_stage_from_ckpt(first_stage_config)
        self.inr = instantiate_from_config(config=inr_config)
        del self.inr.decoder
        del self.inr.post_quant_conv
        del self.inr.quantize
        self.cond_list = inr_config['params']['metadata']['encoder_params']['params']['cond_list']
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor

        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))
        self.perceptual_loss = LPIPS().eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self, batch, return_logits=False):
        # one step to produce the logits
        x_gt = batch[self.first_stage_key] ##"ctslice"
        _, z_indices = self.encode_to_z_first_stage(x_gt)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        batch['image_key'] = self.first_stage_key
        logits, qloss, quant = self.encode_to_z_inr(batch, return_logits)
        return logits, target, qloss, quant

    def encode_to_z_inr(self, batch, return_logits):
        h = self.inr.encode_to_prequant(batch)
        if return_logits:
            quant, qloss, info, logits = self.first_stage_model.quantize(h, return_logits=return_logits)
            return logits, qloss, quant
        else:
            quant, qloss, info = self.first_stage_model.quantize(h, return_logits=return_logits)
            return quant, qloss, info

    @torch.no_grad()
    def encode_to_z_first_stage(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        #indices = self.permuter(indices)
        return quant_z, indices

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

    @rank_zero_only
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step == 0:
            losses = ''
            for k, v in loss_dict.items():
                split, loss_name = k.split("/")
                losses = f"{losses}, {loss_name} : {float(v):.3f}"
            losses = f"[{split} Step{self.global_step}] : {losses[2:]}"
            print(f"{losses}")

    @torch.no_grad()
    def calculate_psnr(self, batch, dec_input):
        x = batch[self.first_stage_key].to(self.device)
        xrec = self.first_stage_model.decode(dec_input)
        return self.psnr(xrec, x), self.perceptual_loss(x.contiguous(), xrec.contiguous()).detach().mean()

    def shared_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        logits, target, qloss, quant = self(batch, True)
        logits = rearrange(logits, 'b c h w -> b (h w) c')
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        loss = F.cross_entropy(logits, target)
        psnr, ploss = self.calculate_psnr(batch, quant)
        return loss, qloss, psnr, ploss

    def training_step(self, batch, batch_idx):
        # print("train")
        # pdb.set_trace()
        loss, qloss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"train/quant_loss": qloss, "train/psnr": psnr, "train/p_loss": ploss}
        self.log("train/ce_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        log_dict['train/ce_loss'] = loss
        self.print_loss(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("validation")
        # pdb.set_trace()
        loss, qloss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"val/quant_loss": qloss, "val/psnr": psnr, "val/p_loss": ploss}
        log_dict['val/ce_loss'] = loss
        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.print_loss(log_dict)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0.9, 0.95))
        #optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0, 0.9))
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.first_stage_key
        x_gt = batch[self.first_stage_key]
        x_gt = x_gt.to(self.device)

        # encode
        quant, _, info = self.encode_to_z_inr(batch, return_logits=False)
        # decode
        x_rec = self.first_stage_model.decode(quant)
        x_rec_first = self.first_stage_model(x_gt)[0]

        for k, v in batch.items():
            if k in self.cond_list:
                log[k] = v

        log["inputs"] = x_gt
        log["decoded"] = x_rec_first
        log["reconstructions"] = x_rec
        return log

class Net2NetINRVQ(pl.LightningModule):
    def __init__(self,
                 inr_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 metadata={}
                 ):
        super().__init__()
        self.print_loss_per_step = metadata.get('print_loss_per_step', 50)
        self.first_stage_key = first_stage_key  ## ctslice
        self.first_stage_n_embed = first_stage_config['params']['n_embed']
        self.init_first_stage_from_ckpt(first_stage_config)
        self.inr = instantiate_from_config(config=inr_config)
        del self.inr.decoder
        del self.inr.post_quant_conv
        del self.inr.quantize
        self.cond_list = inr_config['params']['metadata']['encoder_params']['params']['cond_list']
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor

        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))
        self.perceptual_loss = LPIPS().eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self, batch, return_logits=False):
        # one step to produce the logits
        x_gt = batch[self.first_stage_key] ##"ctslice"
        _, z_indices = self.encode_to_z_first_stage(x_gt)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        batch['image_key'] = self.first_stage_key
        quant, qloss, logits_index = self.encode_to_z_inr(batch, return_logits)
        return target, quant, qloss, logits_index

    def encode_to_z_inr(self, batch, return_logits):
        z = self.inr.encode_to_prequant(batch)
        quant, qloss, info = self.first_stage_model.quantize(z)
        if return_logits:   ## For train
            z = rearrange(z, 'b c h w -> b h w c').contiguous()
            z_flattened = z.view(-1, self.first_stage_model.quantize.e_dim)
            embedding = self.first_stage_model.quantize.embedding
            # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(embedding.weight, 'n d -> d n'))
            logits = torch.sum(d, dim=1, keepdim=True) - d

            return quant, qloss, logits
        else:       ## For test or visualize
            return quant, qloss, info[2]

    @torch.no_grad()
    def encode_to_z_first_stage(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        #indices = self.permuter(indices)
        return quant_z, indices

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

    @rank_zero_only
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step == 0:
            losses = ''
            for k, v in loss_dict.items():
                split, loss_name = k.split("/")
                losses = f"{losses}, {loss_name} : {float(v):.3f}"
            losses = f"[{split} Step{self.global_step}] : {losses[2:]}"
            print(f"{losses}")

    @torch.no_grad()
    def calculate_psnr(self, batch, dec_input):
        x = batch[self.first_stage_key].to(self.device)
        xrec = self.first_stage_model.decode(dec_input)
        return self.psnr(xrec, x), self.perceptual_loss(x.contiguous(), xrec.contiguous()).detach().mean()

    def shared_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        target, quant, qloss, logits = self(batch, True)
        # logits = rearrange(logits, 'b c h w -> b (h w) c')
        # logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        loss = F.cross_entropy(logits, target)
        psnr, ploss = self.calculate_psnr(batch, quant)
        return loss, qloss, psnr, ploss

    def training_step(self, batch, batch_idx):
        # print("train")
        # pdb.set_trace()
        loss, qloss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"train/quant_loss": qloss, "train/psnr": psnr, "train/p_loss": ploss}
        self.log("train/ce_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        log_dict['train/ce_loss'] = loss
        self.print_loss(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        # print("validation")
        # pdb.set_trace()
        loss, qloss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"val/quant_loss": qloss, "val/psnr": psnr, "val/p_loss": ploss}
        log_dict['val/ce_loss'] = loss
        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.print_loss(log_dict)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0.9, 0.95))
        #optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0, 0.9))
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.first_stage_key
        x_gt = batch[self.first_stage_key]
        x_gt = x_gt.to(self.device)

        # encode
        quant, _, _ = self.encode_to_z_inr(batch, return_logits=False)
        # decode
        x_rec = self.first_stage_model.decode(quant)
        x_rec_first = self.first_stage_model(x_gt)[0]

        for k, v in batch.items():
            if k in self.cond_list:
                log[k] = v

        log["inputs"] = x_gt
        log["decoded"] = x_rec_first
        log["reconstructions"] = x_rec
        return log


class Net2NetINRAE(pl.LightningModule):
    def __init__(self,
                 inr_config,
                 first_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 monitor=None,
                 metadata={}
                 ):
        super().__init__()
        self.print_loss_per_step = metadata.get('print_loss_per_step', 50)
        self.first_stage_key = first_stage_key  ## ctslice
        self.first_stage_n_embed = first_stage_config['params']['n_embed']
        self.init_first_stage_from_ckpt(first_stage_config)
        self.inr = instantiate_from_config(config=inr_config)
        del self.inr.decoder
        self.cond_list = inr_config['params']['metadata']['encoder_params']['params']['cond_list']
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
            self.monitor = monitor
        self.psnr = lambda recon, gt: mse2psnr(img2mse(recon.contiguous(), gt.contiguous()))
        self.perceptual_loss = LPIPS().eval()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def forward(self, batch):
        # one step to produce the logits
        x_gt = batch[self.first_stage_key] ##"ctslice"
        target_feature = self.encode_to_z_first_stage(x_gt)

        # make the prediction
        batch['image_key'] = self.first_stage_key
        feature = self.encode_to_z_inr(batch)

        return feature, target_feature

    def encode_to_z_inr(self, batch):
        feature = self.inr.encode_to_prequant(batch)
        return feature

    @torch.no_grad()
    def encode_to_z_first_stage(self, x):
        feature = self.first_stage_model.encode(x)
        return feature

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

    @rank_zero_only
    def print_loss(self, loss_dict):
        if self.global_step % self.print_loss_per_step == 0:
            losses = ''
            for k, v in loss_dict.items():
                split, loss_name = k.split("/")
                losses = f"{losses}, {loss_name} : {float(v):.3f}"
            losses = f"[{split} Step{self.global_step}] : {losses[2:]}"
            print(f"{losses}")

    @torch.no_grad()
    def calculate_psnr(self, batch, dec_input):
        x = batch[self.first_stage_key].to(self.device)
        xrec = self.first_stage_model.decode(dec_input)
        return self.psnr(xrec, x), self.perceptual_loss(x.contiguous(), xrec.contiguous()).detach().mean()

    def shared_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        feature, target_feature = self(batch)
        loss = F.mse_loss(feature, target_feature)
        psnr, ploss = self.calculate_psnr(batch, feature)
        return loss, psnr, ploss

    def training_step(self, batch, batch_idx):
        loss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"train/psnr": psnr, "train/p_loss": ploss}
        self.log("train/feat_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        log_dict.update({'train/feat_loss': loss})
        self.print_loss(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, psnr, ploss = self.shared_step(batch, batch_idx)
        log_dict = {"val/psnr": psnr, "val/p_loss": ploss}
        log_dict.update({'val/feat_loss': loss})
        self.log(self.monitor, log_dict[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log_dict(log_dict)
        self.print_loss(log_dict)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0.9, 0.95))
        #optimizer = torch.optim.Adam(list(self.inr.parameters()), lr=lr, betas=(0, 0.9))
        return optimizer

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.first_stage_key
        x_gt = batch[self.first_stage_key]
        x_gt = x_gt.to(self.device)

        # encode
        feature = self.encode_to_z_inr(batch)
        # decode
        x_rec = self.first_stage_model.decode(feature)
        x_rec_first = self.first_stage_model(x_gt)

        for k, v in batch.items():
            if k in self.cond_list:
                log[k] = v

        log["inputs"] = x_gt
        log["decoded"] = x_rec_first
        log["reconstructions"] = x_rec
        return log