import os
import sys
import pdb
import torch
import torchvision
import torch.nn as nn

sys.path.append(os.getcwd())

from collections import OrderedDict
from x2ct_nerf.modules.image_encoder.get_transform import get_data_transform


def update_pretrained_weight(model, pretrained_weight):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weight.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def set_model(pretrained_weight, weight_dir, model_name):
    print(f"model name : {model_name}")
    if pretrained_weight == "imagenet":
        model = getattr(torchvision.models, model_name)(pretrained=True)
    elif pretrained_weight == "autoenc_LIDC":
        model = getattr(torchvision.models, model_name)(pretrained=False)
        pretrained_weight = torch.load(weight_dir)
        model = update_pretrained_weight(model, pretrained_weight["network_state_dict"])
    elif pretrained_weight is None:
        model = getattr(torchvision.models, model_name)(pretrained=False)
        print("Use RenNet34 Image Encoder from the scratch")
    else:
        raise NotImplementedError
    return model


class ResNetEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResNetEncoder, self).__init__()
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        self.args_encoder = cfg
        self.in_channels = cfg["in_channels"]
        self.latent_dim = cfg["latent_dim"]
        self.input_img_size = cfg["input_img_size"]
        self.encoder_freeze_layer = cfg["encoder_freeze_layer"]
        self.feature_layer = cfg["feature_layer"]
        self.pretrained = cfg["pretrained"]
        self.weight_dir = cfg["weight_dir"]
        self.autocast = cfg.get("autocast", False)
        model_name = cfg.get("model_name", "resnet34")

        assert not ((self.in_channels == 1) and self.encoder_freeze_layer)
        assert self.feature_layer in ["layer1", "layer2", "layer3", "layer4", "all"]
        assert self.pretrained in [None, "autoenc_LIDC", "imagenet"]

        self.model = set_model(pretrained_weight=self.pretrained, weight_dir=self.weight_dir, model_name=model_name)
        self.transform = get_data_transform(in_channels=self.in_channels, input_img_size=self.input_img_size)

        if self.in_channels != 3:
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        if self.feature_layer != "all":
            del self.model.fc
            del self.model.avgpool
            for layer in ["layer1", "layer2", "layer3", "layer4"][::-1]:
                if layer == self.feature_layer:
                    break
                delattr(self.model, layer)

            tmp = [n for n in getattr(self.model, self.feature_layer)[-1].named_modules() if len(n[0]) > 0]
            for name, layer in tmp[::-1]:
                if name[:4] == 'conv':
                    output_dim =layer.out_channels
                    break

            if output_dim != self.latent_dim:
                self.conv2 = nn.Conv2d(output_dim, self.latent_dim, kernel_size=(3, 3), padding=1, bias=False)  ## batch x latent_dim x 10 x 10
                self.bn2 = nn.BatchNorm2d(self.latent_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                self.relu = nn.ReLU()
        else:
            output_dim = self.model.fc.out_features
            if output_dim != self.latent_dim:
                self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.latent_dim)

        if self.pretrained and self.encoder_freeze_layer:
            for name, param in self.model.named_parameters():
                if self.encoder_freeze_layer in name:
                    break
                param.requires_grad = False

        self.output_dim = output_dim
        self.output_ch = self.latent_dim

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_dim)
        """
        if self.in_channels == 3 and x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        x = self.transform(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        # for input size (3, 320, 320)
        # layer1: (64, 80, 80), layer2: (128, 40, 40), layer3: (256, 20, 20), layer4: (512, 10, 10)
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self.model, layer)(x)
            if layer == self.feature_layer:
                break

        if self.feature_layer != "all":
            if self.output_dim != self.output_ch:
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)

        if self.autocast:
            return x
        else:
            return x.float()


class ResNetUNetEncoder(ResNetEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.zero_test = cfg.get("zero_test", False)

    def forward(self, x):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, latent_dim)
        """
        # B, 3, 128, 128
        if self.in_channels == 3 and x.shape[1] == 1:
            x = torch.cat([x] * 3, dim=1)
        
        skipfeatures = {}
        x = self.transform(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        skipfeatures['conv'] = x.clone()
        
        x = self.model.maxpool(x) 

        # for input size (3, 128, 128), resnet34
        # conv: (64, 64, 64) / (64, 32, 32), layer1: (64, 32, 32), layer2: (128, 16, 16)

        # for input size (3, 128, 128), resnet101
        # conv(before/after maxpool): (64, 64, 64) / (64, 32, 32), layer1: (256, 32, 32), layer2: (512, 16, 16)
        
        for layer in ["layer1", "layer2", "layer3", "layer4"]:
            x = getattr(self.model, layer)(x) 
            if layer == 'layer1':
                skipfeatures[layer] = x.clone()
            if layer == self.feature_layer:
                break

        if self.feature_layer != "all":
            if self.output_dim != self.output_ch:
                x = self.conv2(x)
                x = self.bn2(x)
        else:
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.model.fc(x)

        if self.autocast:
            return x, skipfeatures
        else:
            # if self.zero_test:
            #     skipfeatures = {k:torch.zeros_like(v) for k, v in skipfeatures.items()}
            # else:
            skipfeatures = {k:v.float() for k, v in skipfeatures.items()}
            return x.float(), skipfeatures


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        features = in_channels // 16
        self.upconv5 = nn.ConvTranspose2d(in_channels, features * 16, kernel_size=2, stride=2)
        self.decoder5 = Decoder._block(features * 16, features * 16, name="dec4")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Decoder._block(features * 8, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Decoder._block(features * 4, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Decoder._block(features * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Decoder._block(features, features, name="dec1")
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)  # (64, output_channel, 12, stride=4, padding=4)

    def forward(self, x):
        dec5 = self.upconv5(x)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        x = self.conv(dec1)
        x = torch.sigmoid(x)
        return x

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class ResNetAutoEncoder(nn.Module):
    def __init__(self, pretrained=True, in_channels=3, out_channels=1, latent_dim=128, input_img_size=320, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = ResNetEncoder(pretrained=pretrained, in_channels=in_channels, latent_dim=latent_dim, input_img_size=input_img_size)
        self.decoder = Decoder(in_channels=latent_dim, out_channels=out_channels)
        self.transform = get_data_transform(in_channels=in_channels, input_img_size=input_img_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import torch

    # self.args_encoder = cfg
    #     self.in_channels = cfg["in_channels"]
    #     self.latent_dim = cfg["latent_dim"]
    #     self.input_img_size = cfg["input_img_size"]
    #     self.encoder_freeze_layer = cfg["encoder_freeze_layer"]
    #     self.feature_layer = cfg["feature_layer"]
    #     self.pretrained = cfg["pretrained"]
    #     self.weight_dir = cfg["weight_dir"]
    #     self.autocast = cfg.get('autocast', False)a
    cfg = {
        "model_name": 'resnet50',
        "in_channels": 3,
        "latent_dim": 128,
        "input_img_size": 128,
        "encoder_freeze_layer": "layer1",
        "pretrained": "imagenet",
        "weight_dir": None,
        "feature_layer": "all",
        "skipconnect_layer": ('conv', 'layer1', 'layer2', 'layer3', 'layer4')
    }
    x = torch.randn((2, 3, 128, 128))
    model = ResNetUNetEncoder(cfg)
    # print(model)
    output = model(x)
    print(output.shape)
    # model = ResNetAutoEncoder(latent_dim=512)
    # output = model(x)
    # print(output.shape)
