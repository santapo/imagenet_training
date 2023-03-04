from timm.models.efficientnet import efficientnetv2_s
from timm.models.mlp_mixer import (mixer_b16_224, mixer_b32_224, mixer_s16_224,
                                   mixer_s32_224)
from timm.models.vision_transformer import (vit_base_patch16_224,
                                            vit_base_patch32_224,
                                            vit_small_patch16_224,
                                            vit_small_patch32_224)
from torchvision.models.resnet import (resnet18, resnet34, resnet50, resnet101,
                                       resnet152)

_model_factory = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "efficientnetv2_s": efficientnetv2_s,
    "vit-s-16": vit_small_patch16_224,
    "vit-s-32": vit_small_patch32_224,
    "vit-b-16": vit_base_patch16_224,
    "vit-b-32": vit_base_patch32_224,
    "mixer-s-32": mixer_s32_224,
    "mixer-s-16": mixer_s16_224,
    "mixer-b-16": mixer_b16_224,
    "mixer-b-32": mixer_b32_224,
}


def get_model(name: str, num_classes: int, pretrained: bool):
    try:
        model = _model_factory[name](num_classes=num_classes, pretrained=pretrained)
    except KeyError:
        raise ValueError(f"Model {name} is not supported")
    return model