import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from importlib import import_module
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision.transforms as transforms
import pdb

from main import instantiate_from_config
from utils.metrics import mse2psnr, img2mse
from collections import OrderedDict
from x2ct_nerf.modules.utils import set_requires_grad


class AEModel(pl.LightningModule):
    """
    without codebook, Use CT Only, Use original taming modules
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 metadata={},
                 ):
        super().__init__()
        self.metadata = metadata
        self.image_key = image_key
        self.print_loss_per_step = self.metadata.get('print_loss_per_step', 50)
        encoder_module, decoder_module = self.get_encoder_decoder_module()
        encoder_params = self.metadata['encoder_params'] if 'encoder_params' in self.metadata.keys() else ddconfig
        decoder_params = self.metadata['decoder_params'] if 'decoder_params' in self.metadata.keys() else ddconfig
        self.encoder = encoder_module(**encoder_params)
        self.decoder = decoder_module(**decoder_params)
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

    def get_encoder_decoder_module(self):
        encoder_module = self.metadata['encoder_module'] if 'encoder_module' in self.metadata.keys() else 'taming.modules.diffusionmodules.model.Encoder'
        encoder_module, enc_class = encoder_module.rsplit(".", 1)
        encoder_module = getattr(import_module(encoder_module), enc_class)

        decoder_module = self.metadata['decoder_module'] if 'decoder_module' in self.metadata.keys() else 'taming.modules.diffusionmodules.model.Decoder'
        decoder_module, dec_class = decoder_module.rsplit(".", 1)
        decoder_module = getattr(import_module(decoder_module), dec_class)
        return encoder_module, decoder_module

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

    def encode(self, x):
        h = self.encoder(x)
        return h

    def decode(self, feature):
        dec = self.decoder(feature)
        return dec

    def decode_code(self, code_b):
        pass

    def forward(self, input):
        feature = self.encode(input)
        assert not isinstance(feature, dict)
        dec = self.decode(feature)
        output_dict = {
            'outputs': dec,
        }
        return output_dict

    def get_additional_input(self, batch, keys):
        xs = {}
        for k, x in batch.items():
            if k in keys:
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

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
        qloss = torch.zeros(len(x), 1).to(self.device)
        xrec_dict = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict["outputs"], x)
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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        qloss = torch.zeros(len(x), 1).to(self.device)
        xrec_dict = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        try:
            return self.decoder.conv_out.weight
        except:
            return None
        

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec = self(x)['outputs']
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


class INRAEModel(AEModel):
    """
    without codebook, Use CT, Xray
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 metadata={},
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         metadata=metadata,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed
        self.use_quant_conv = metadata.get("use_quant_conv", False)
        if self.use_quant_conv:
            self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

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

    def forward(self, input):
        feature = self.encode(input)
        feature = feature['outputs']
        if self.use_quant_conv:
            feature = self.quant_conv(feature)
        dec = self.decode(feature)
        output_dict = {
            'outputs': dec,
        }
        return output_dict

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        return h

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict["outputs"], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #print("validation step")
        # pdb.set_trace()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        # pdb.set_trace()
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
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
        if 'p0' in kwargs.keys() and 'zoom_size' in kwargs.keys():
            h = self.encoder(batch, p0=kwargs['p0'], zoom_size=kwargs['zoom_size'])
        else:
            h = self.encoder(batch)

        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        # quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(h['outputs'])

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class INRAEModelMultiDecoderOut(INRAEModel):
    """
    without codebook, Use CT, Xray
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 metadata={},
                 ):

        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path,
                         ignore_keys,
                         image_key,
                         colorize_nlabels,
                         monitor,
                         kl_weight,
                         remap,
                         metadata,
                         )
        self.output_key = self.metadata['decoder_params']['output_key']
        self.output_key.append('final')
        self.low2high_for_disc = self.metadata['decoder_params']['low2high_for_disc']

    def get_last_layer(self, key):
        assert key in ['mid', 'up_block0', 'final']
        if key == 'mid':
            return self.decoder.mid.block_2.conv2.weight
        elif key == 'up_block0':
            return self.decoder.up[2].upsample.conv.weight
        elif key == 'final':
            return self.decoder.conv_out.weight

    def forward(self, input):
        feature = self.encode(input)
        feature = feature['outputs']
        if self.use_quant_conv:
            feature = self.quant_conv(feature)
        output_dict = self.decode(feature)
        output_dict_dict = {}
        for k in self.output_key:
            v = output_dict[k][:, 0:1].repeat(1, 3, 1, 1) if k != 'final' else output_dict[k]
            output_dict_dict[k] = {'outputs': v}

        return output_dict_dict

    def shared_step(self, batch, split, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key

        output_dict_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)

        if optimizer_idx == 0:
            log_dict_ae = {}
            aeloss = None
            for output_k, xrec_dict in output_dict_dict.items():
                if xrec_dict['outputs'].shape[-1] != x.shape[-1]:
                    resized_x = F.interpolate(x, size=xrec_dict['outputs'].shape[-1], mode="bilinear")
                else:
                    resized_x = x
                aeloss_, log_dict_ae_ = self.loss(qloss, resized_x, xrec_dict, optimizer_idx, self.global_step,
                                                  last_layer=self.get_last_layer(output_k), split=split)
                aeloss = aeloss_ if aeloss is None else aeloss + aeloss_
                if output_k != 'final':
                    for k, v in log_dict_ae_.items():
                        log_dict_ae[f"{k}_{output_k}"] = v
                else:
                    log_dict_ae.update(log_dict_ae_)
            log_dict_ae[f'{split}/psnr'] = self.psnr(output_dict_dict['final']["outputs"], x)
            return aeloss, log_dict_ae

        elif optimizer_idx == 1:
            # discriminator
            if self.low2high_for_disc:
                resized_xs = None
                resized_recs = None
                for output_k, xrec_dict in output_dict_dict.items():
                    if xrec_dict['outputs'].shape[-1] != x.shape[-1]:
                        resized_x = F.interpolate(x, size=xrec_dict['outputs'].shape[-1], mode="bilinear")
                    else:
                        resized_x = x
                    resized_x = F.interpolate(resized_x, size=x.shape[-1], mode="bilinear")[:, 0:1]
                    resized_rec = F.interpolate(xrec_dict['outputs'], size=x.shape[-1], mode="bilinear")[:, 0:1]
                    resized_xs = torch.cat((resized_xs, resized_x), dim=1) if resized_xs is not None else resized_x
                    resized_recs = torch.cat((resized_recs, resized_rec),
                                             dim=1) if resized_recs is not None else resized_rec
            else:
                resized_xs = x
                resized_recs = output_dict_dict['final']['outputs']

            discloss, log_dict_disc = self.loss(qloss, resized_xs, {"outputs": resized_recs}, optimizer_idx,
                                                self.global_step, last_layer=self.get_last_layer('final'), split=split)

            return discloss, log_dict_disc

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            aeloss, log_dict_ae = self.shared_step(batch, 'train', optimizer_idx)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            discloss, log_dict_disc = self.shared_step(batch, 'train', optimizer_idx)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        aeloss, log_dict_ae = self.shared_step(batch, 'val', 0)
        discloss, log_dict_disc = self.shared_step(batch, 'val', 1)
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
        h = self.encoder(batch)
        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])

        # decode
        output_dict = self.decode(h['outputs'])
        output_dict_dict = {}
        for k in self.output_key:
            v = output_dict[k][:, 0:1].repeat(1, 3, 1, 1) if k != 'final' else output_dict[k]
            output_dict_dict[k] = {'outputs': v}

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        for output_k, xrec_dict in output_dict_dict.items():
            if xrec_dict['outputs'].shape[-1] != x.shape[-1]:
                resized_x = F.interpolate(x, size=xrec_dict['outputs'].shape[-1], mode="bilinear")
            else:
                resized_x = x
            resized_x = F.interpolate(resized_x, size=x.shape[-1], mode="bilinear")
            resized_rec = F.interpolate(xrec_dict['outputs'], size=x.shape[-1], mode="bilinear")
            if output_k == 'final':
                log[f'inputs'] = resized_x
                log[f'reconstructions'] = resized_rec
            else:
                log[f'inputs_{output_k}'] = resized_x
                log[f'reconstructions_{output_k}'] = resized_rec

        return log


class INRAEUNetModel(INRAEModel):
    """
    without codebook, Use CT, Xray
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 metadata={},
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         metadata=metadata,
                         )
        self.feature_res_list = self.metadata['decoder_params'].get("skip_feature_res", None)

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

    def encode(self, x):
        h = self.encoder(x, feature_res_list=self.feature_res_list)
        return h

    def decode(self, feature, skipfeatures):
        dec = self.decoder(feature, skipfeatures)['final']
        return dec

    def forward(self, input):
        feature = self.encode(input)
        enc = feature['outputs']
        skipfeatures = feature['skipfeatures']
        if self.use_quant_conv:
            enc = self.quant_conv(enc)
        dec = self.decode(enc, skipfeatures)
        output_dict = {
            'outputs': dec,
        }
        return output_dict

    def encode_to_prequant(self, x):
        h = self.encoder(x, feature_res_list=self.feature_res_list)
        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        return h

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict["outputs"], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
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
        if 'p0' in kwargs.keys() and 'zoom_size' in kwargs.keys():
            h = self.encoder(batch, p0=kwargs['p0'], zoom_size=kwargs['zoom_size'])
        elif self.feature_res_list is not None:
            h = self.encoder(batch, feature_res_list=self.feature_res_list)
        else:
            h = self.encoder(batch)

        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])

        x_rec = self.decode(h['outputs'], h['skipfeatures'])

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class INRAETemplateModel(AEModel):
    """
    without codebook, Use CT, Xray
    """

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 metadata={},
                 ):
        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         metadata=metadata,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed
        self.use_quant_conv = metadata.get("use_quant_conv", False)
        self.decoder_freeze = metadata.get("decoder_freeze", False)
        self.base_constraint_option = metadata.get("base_constraint_option", None)

        if self.use_quant_conv:
            self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        if metadata.get("decoder_weight_dir", None) is not None:
            decode_weight_dict = OrderedDict()
            decode_weight = torch.load(metadata["decoder_weight_dir"])["state_dict"]

            for k, v in decode_weight.items():
                if 'decoder' in k:
                    decode_weight_dict[k.replace("decoder.", "")] = v

            self.decoder.load_state_dict(decode_weight_dict)

            if self.decoder_freeze:
                set_requires_grad(self.decoder, False)

    def get_input(self, batch):
        xs = {}
        for k, x in batch.items():
            if k in ['ctslice', 'PA', 'Lateral', 'base_PA', 'base_Lateral']:
                if len(x.shape) == 3:
                    x = x[..., None]
                x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

    def forward(self, input):
        feature = self.encode(input)
        others = {k: feature[k] for k in feature if k != 'outputs'}
        feature = feature['outputs']
        if self.use_quant_conv:
            feature = self.quant_conv(feature)
        dec = self.decode(feature)
        output_dict = {
            'outputs': dec,
            'dx': others['dx'],
            'dsigma': others['dsigma'],
        }
        return output_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch = self.setting_base(batch)
        batch['image_key'] = self.image_key
        xrec = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        if optimizer_idx == 0:
            # autoencode
            base_mask = batch.get('base_mask', None)
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", base_mask=base_mask)
            log_dict_ae['train/psnr'] = self.psnr(xrec["outputs"], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.get_input(batch)
        batch = self.setting_base_val(batch)
        batch['image_key'] = self.image_key
        xrec = self(batch)
        x = batch[self.image_key]
        qloss = torch.zeros(len(x), 1).to(self.device)
        base_mask = batch.get('base_mask', None)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val", base_mask=base_mask)

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        log_dict_ae['val/psnr'] = self.psnr(xrec["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        if 'base_file_path_' not in batch and kwargs['split'] == 'train':
            batch = self.setting_base(batch)
        elif kwargs['split'] == 'val':
            batch = self.setting_base_val(batch)
        else:
            raise NotImplementedError
        batch['image_key'] = self.image_key
        x = batch[self.image_key]

        x = x.to(self.device)
        # encode
        h = self.encoder(batch)
        dx = h['dx'].clone()
        dsigma = h['dsigma'].clone()
        h = h['outputs']
        if self.use_quant_conv:
            h = self.quant_conv(h)
        x_rec = self.decode(h)

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        log["dx"] = dx.reshape(x.shape[0], self.metadata['encoder_params']['params']['feature_res'], self.metadata['encoder_params']['params']['feature_res'], -1).mean(-1)
        log["dx"] = F.interpolate(torch.stack([log["dx"]] * 3, dim=1), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        log["dx"] = (log["dx"] - log["dx"].min()) / (log["dx"].max() - log["dx"].min())

        log["dsigma"] = dsigma.reshape(x.shape[0], self.metadata['encoder_params']['params']['feature_res'], self.metadata['encoder_params']['params']['feature_res'], -1).mean(-1)
        log["dsigma"] = F.interpolate(torch.stack([log["dsigma"]] * 3, dim=1), size=(x.shape[-2], x.shape[-1]), mode='nearest')
        log["dsigma"] = (log["dsigma"] - log["dsigma"].min()) / (log["dsigma"].max() - log["dsigma"].min())
        return log

    def setting_base(self, batch):
        assert batch['ctslice'].shape[0] % 2 == 0
        _batch_size = batch['ctslice'].shape[0] // 2
        _equal_base = int(_batch_size * self.base_constraint_option)
        key_list = list(batch.keys())
        for k in key_list:
            if k != 'ctslice':
                batch['base_' + k] = batch[k][_batch_size:-_equal_base]
                if k == 'file_path_':
                    batch['base_' + k].extend(batch[k][_batch_size-_equal_base:_batch_size])
                else:
                    batch['base_' + k] = torch.cat([batch['base_' + k], batch[k][_batch_size-_equal_base:_batch_size]], dim=0)
            batch[k] = batch[k][:_batch_size]
        batch['base_mask'] = [1 if p1.split('/')[-3] == p2.split('/')[-3] else 0 for p1, p2 in zip(batch['base_file_path_'], batch['file_path_'])]
        batch['base_mask'] = torch.tensor(batch['base_mask'], device=batch['ctslice'].device).bool()
        self.encoder.network_fn.base_mask = batch['base_mask']
        return batch

    def setting_base_val(self, batch):
        batch['base_mask'] = [1 if p1.split('/')[-3] == p2.split('/')[-3] else 0 for p1, p2 in zip(batch['base_file_path_'], batch['file_path_'])]
        batch['base_mask'] = torch.tensor(batch['base_mask'], device=batch['ctslice'].device).bool()
        self.encoder.network_fn.base_mask = batch['base_mask']
        return batch
