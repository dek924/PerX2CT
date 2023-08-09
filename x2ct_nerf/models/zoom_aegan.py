import torch
from torchvision.transforms import Resize
import pdb

from x2ct_nerf.models.aegan import AEModel


class INRAEZoomModel(AEModel):
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
        self.gt_key = metadata.get("gt_key")
        print(f"gt_key : {self.gt_key}")
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
            elif k in ['ct']:
                if len(x.shape) == 4:
                    x = x[..., None]
                x = x.permute(0, 4, 1, 2, 3).to(memory_format=torch.contiguous_format)
                xs[k] = x.float().to(self.device)
            else:
                xs[k] = x
        return xs

    def forward(self, input):
        feature = self.encode(input)
        if self.gt_key == self.image_key:
            x = input[self.image_key]
        else:
            x = feature[self.gt_key]

        feature = feature['outputs']

        if self.use_quant_conv:
            feature = self.quant_conv(feature)
        dec = self.decode(feature)
        if dec.shape[1] == 1:
            dec = torch.cat([dec] * 3, dim=1)
        output_dict = {
            'outputs': dec,
        }
        return output_dict, x

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        return h

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict, x = self(batch)

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
        xrec_dict, x = self(batch)
        qloss = torch.zeros(len(x), 1).to(self.device)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
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
        if x_rec.shape[1] == 1:
            x_rec = torch.cat([x_rec] * 3, dim=1)
        input_resolution = x_rec.shape[-1]

        # input conditions
        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = torch.cat((x[:, 1:2], x[:, 1:2], x[:, 1:2]), dim=1)

        if self.gt_key in h.keys():
            crop_ct = h[self.gt_key]
            log[self.gt_key] = torch.cat((crop_ct[:, 1:2], crop_ct[:, 1:2], crop_ct[:, 1:2]), dim=1)

        log["reconstructions"] = torch.cat((x_rec[:, 1:2].clone(), x_rec[:, 1:2].clone(), x_rec[:, 1:2].clone()), dim=1)

        for k, v in log.items():
            if v.shape[-1] != input_resolution:
                log[k] = Resize(input_resolution)(v)
        return log


class INRAEZoomSkipConnectModel(AEModel):
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
        self.gt_key = metadata.get("gt_key")
        self.feature_res_list = self.metadata['decoder_params'].get("skip_feature_res", None)
        print(f"gt_key : {self.gt_key}")
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

    def encode(self, x):
        h = self.encoder(x, feature_res_list=self.feature_res_list)
        return h

    def decode(self, feature, skipfeatures):
        dec = self.decoder(feature, skipfeatures)['final']
        return dec

    def forward(self, input):
        feature = self.encode(input)
        if self.gt_key == self.image_key:
            x = input[self.image_key]
        else:
            x = feature[self.gt_key]

        skipfeatures = feature['skipfeatures']
        feature = feature['outputs']

        if self.use_quant_conv:
            feature = self.quant_conv(feature)

        dec = self.decode(feature, skipfeatures)
        output_dict = {
            'outputs': dec,
        }
        return output_dict, x

    def encode_to_prequant(self, x):
        h = self.encoder(x, feature_res_list=self.feature_res_list)
        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        return h

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.get_input(batch)
        batch['image_key'] = self.image_key
        xrec_dict, x = self(batch)

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
        xrec_dict, x = self(batch)
        qloss = torch.zeros(len(x), 1).to(self.device)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec_dict, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec_dict, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        log_dict_ae['val/psnr'] = self.psnr(xrec_dict["outputs"], x)
        self.log(self.monitor, log_dict_ae[self.monitor], prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        #self.log("val/aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
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
            h = self.encoder(batch, feature_res_list=self.feature_res_list, p0=kwargs['p0'], zoom_size=kwargs['zoom_size'])
        elif self.feature_res_list is not None:
            h = self.encoder(batch, feature_res_list=self.feature_res_list)
        else:
            h = self.encoder(batch)

        if self.use_quant_conv:
            h['outputs'] = self.quant_conv(h['outputs'])
        # quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(h['outputs'], h['skipfeatures'])
        input_resolution = x_rec.shape[-1]

        # input conditions
        for k, v in batch.items():
            if k in self.metadata['encoder_params']['params']['cond_list']:
                log[k] = v

        log["inputs"] = torch.cat((x[:, 1:2], x[:, 1:2], x[:, 1:2]), dim=1)

        if self.gt_key in h.keys():
            crop_ct = h[self.gt_key]
            log[self.gt_key] = torch.cat((crop_ct[:, 1:2], crop_ct[:, 1:2], crop_ct[:, 1:2]), dim=1)

        log["reconstructions"] = torch.cat((x_rec[:, 1:2], x_rec[:, 1:2], x_rec[:, 1:2]), dim=1)

        for k, v in log.items():
            if v.shape[-1] != input_resolution:
                log[k] = Resize(input_resolution)(v)
        return log
