import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from importlib import import_module
from pytorch_lightning.utilities.distributed import rank_zero_only
import pdb

from main import instantiate_from_config

from x2ct_nerf.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from x2ct_nerf.modules.vqvae.quantize import GumbelQuantize
from utils.metrics import mse2psnr, img2mse

class VQModel(pl.LightningModule):
    """
    Use taming module
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
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        #self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
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

        print(f"encoder class : {enc_class}")
        print(f"decoder class : {dec_class}")
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

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        output = {
            'outputs': dec
        }
        return output, diff

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
        xrec_dict, qloss = self(x)

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
        x = self.get_input(batch, self.image_key)
        xrec_dict, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        log_dict_ae['val/psnr'] = self.psnr(xrec_dict['outputs'], x)
        #rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(self.monitor, log_dict_ae[self.monitor],
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        #xrec, _ = self(x)
        xrec = self(x)[0]['outputs']
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


class INRVQ(VQModel):
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
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         remap=remap,
                         sane_index_shape=sane_index_shape,
                         metadata=metadata,
                         )
    ## encoder : ImplicitGenerator3dNoRendering
    def encode_to_prequant(self, x):
        h = self.encoder(x)
        assert isinstance(h, dict)
        h["outputs"] = self.quant_conv(h["outputs"])
        return h["outputs"]

    def encode(self, x):
        h = self.encoder(x)
        h["outputs"] = self.quant_conv(h["outputs"])
        quant, emb_loss, info = self.quantize(h["outputs"])
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    # def decode_code(self, code_b):
    #     quant_b = self.quantize.embed_code(code_b)
    #     dec = self.decode(quant_b)
    #     return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        output = {
            "outputs": dec
        }
        return output, diff

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict, qloss = self(batch)
        x = batch[self.image_key]

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
        batch['image_key'] = self.image_key
        xrec_dict, qloss = self(batch)
        x = batch[self.image_key]

        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        log_dict_ae['val/psnr'] = self.psnr(xrec_dict['outputs'], x)
        # rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(self.monitor, log_dict_ae[self.monitor],
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        x = batch[self.image_key]
        x = x.to(self.device)
        #xrec, _ = self(batch)
        xrec = self(batch)[0]['outputs']
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class GumbelVQ(VQModel):
    """
    Use taming module !!
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
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

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    # def decode_code(self, code_b):
    #     raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        #print("Training step")
        #pdb.set_trace()

        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec_dict, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict['outputs'], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        #print("validation step")
        #pdb.set_trace()
        x = self.get_input(batch, self.image_key)
        #xrec, qloss = self(x, return_pred_indices=True)
        xrec_dict, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict['outputs'], x)
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(self.monitor, log_dict_ae[self.monitor],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class INRGumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
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

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    ## encoder : ImplicitGenerator3dNoRendering
    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h["outputs"] = self.quant_conv(h["outputs"])
        return h

    # def decode_code(self, code_b):
    #     raise NotImplementedError

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

    def training_step(self, batch, batch_idx, optimizer_idx):
        #print("Training step")
        #pdb.set_trace()

        self.temperature_scheduling()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict, qloss = self(batch)
        x = batch[self.image_key]

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            log_dict_ae['train/psnr'] = self.psnr(xrec_dict['outputs'], x)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            self.print_loss(log_dict_ae)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        #print("validation step")
        #pdb.set_trace()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        #pdb.set_trace()
        xrec_dict, qloss = self(batch)
        x = batch[self.image_key]
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict['outputs'], x)
        # self.log("val/rec_loss", rec_loss,
        #          prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(self.monitor, log_dict_ae[self.monitor],
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        self.print_loss(log_dict_ae)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        x = batch[self.image_key]

        x = x.to(self.device)
        # encode
        h = self.encoder(batch)
        h['outputs'] = self.quant_conv(h['outputs'])
        quant, _, _ = self.quantize(h['outputs'])
        # decode
        x_rec = self.decode(quant)

        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log