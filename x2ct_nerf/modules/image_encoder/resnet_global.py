import pdb
import copy
from x2ct_nerf.modules.image_encoder.resnet import *

class ResNetGLEncoder(nn.Module):
    def __init__(self, cfg):
        super(ResNetGLEncoder, self).__init__()
        if not isinstance(cfg, dict):
            cfg = vars(cfg)
        self.args_encoder = cfg
        self.in_channels = cfg["in_channels"]
        self.latent_dim = cfg["latent_dim"]
        self.input_img_size = cfg["input_img_size"]
        self.encoder_freeze_layer = cfg["encoder_freeze_layer"]
        self.feature_layer = cfg["feature_layer"]   ## for local
        self.global_feature_layer = cfg["global_feature_layer"]  ## for local
        self.pretrained = cfg["pretrained"]
        self.weight_dir = cfg["weight_dir"]
        model_name = cfg.get("model_name", "resnet34")

        assert not ((self.in_channels == 1) and self.encoder_freeze_layer)
        assert self.feature_layer in ["layer1", "layer2", "layer3", "layer4", "all"]    ## layer2
        assert self.pretrained in [None, "autoenc_LIDC", "imagenet"]                    ## imagenet

        self.model = set_model(pretrained_weight=self.pretrained, weight_dir=self.weight_dir, model_name=model_name)
        self.transform = get_data_transform(in_channels=self.in_channels, input_img_size=self.input_img_size)

        del self.model.fc
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.in_channels != 3:
            self.model.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # delete layers after feature_layer
        for layer in ["layer1", "layer2", "layer3", "layer4"][::-1]:
            if layer == self.global_feature_layer:
                break
            delattr(self.model, layer)

        if 'global_feature_layer_last' in cfg.keys():
            tmp = getattr(self.model, self.global_feature_layer)
            setattr(self.model, self.global_feature_layer, tmp[:cfg['global_feature_layer_last']+1])
            tmp = getattr(self.model, self.global_feature_layer)[-1]
            new_tmp = []
            for name, layer in tmp.named_modules():
                if len(name) > 0 and name in cfg['global_feature_layers_in_last']:
                    new_tmp.append(layer)
            setattr(getattr(self.model, self.global_feature_layer), str(cfg['global_feature_layer_last']), nn.Sequential(*new_tmp))

        ## get feature_layer's dim
        tmp = [n for n in getattr(self.model, self.feature_layer)[-1].named_modules() if len(n[0]) > 0]
        for name, layer in tmp[::-1]:
            if name[:4] == 'conv':
                output_dim = layer.out_channels
                break


        tmp = [n for n in getattr(self.model, self.global_feature_layer)[-1].named_modules() if len(n[0]) > 0]
        for name, layer in tmp[::-1]:
            if isinstance(layer, nn.Conv2d):
                self.global_output_dim = layer.out_channels
                break

        if output_dim != self.latent_dim:
            self.conv2 = nn.Conv2d(output_dim, self.latent_dim, kernel_size=(3, 3), padding=1, bias=False)  ## batch x latent_dim x 10 x 10
            self.bn2 = nn.BatchNorm2d(self.latent_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.relu = nn.ReLU()

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
                local_feature = x
            if layer == self.global_feature_layer:
                global_feature = self.model.avgpool(x)
                break

        if self.output_dim != self.output_ch:
            local_feature = self.conv2(local_feature)
            local_feature = self.bn2(local_feature)
            local_feature = self.relu(local_feature)

        return {'local': local_feature.float(), 'global': global_feature.float()}

