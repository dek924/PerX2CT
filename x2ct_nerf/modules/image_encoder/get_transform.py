import torchvision
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def get_data_transform(in_channels, input_img_size):
    data_aug = []
    data_aug += [torchvision.transforms.Resize(input_img_size)]
    if in_channels == 1:
        data_aug += [torchvision.transforms.Normalize([0.485], [0.229])]
    elif in_channels == 3:
        data_aug += [torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    data_aug = torchvision.transforms.Compose(data_aug)

    return data_aug        