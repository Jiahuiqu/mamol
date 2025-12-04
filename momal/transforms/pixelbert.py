from .utils import (
    inception_normalize,
    MinMaxResize,
)
from torchvision import transforms
from .randaug import RandAugment
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def lidarHS_transform(size):
    # 举例：lidar+HS 拼接后做个简单标准化
    def _t(sample):
        lidar, hs = sample["lidar"], sample["HS"]
        lidar = (lidar - lidar.mean()) / (lidar.std() + 1e-6)
        hs = (hs - hs.mean()) / (hs.std() + 1e-6)
        sample["lidar"] = lidar
        sample["HS"] = hs
        return sample
    return _t
def CLIP_transform(size):
    return Compose([
        Resize(size, interpolation=BICUBIC),
        CenterCrop(size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def pixelbert_transform(size=800):
    longer = int((1333 / 800) * size)
    return transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )


def pixelbert_transform_randaug(size=800):
    longer = int((1333 / 800) * size)
    trs = transforms.Compose(
        [
            MinMaxResize(shorter=size, longer=longer),
            transforms.ToTensor(),
            inception_normalize,
        ]
    )
    trs.transforms.insert(0, RandAugment(2, 9))
    return trs
