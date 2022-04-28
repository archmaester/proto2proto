from .resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features, resnet50_features_inat
from .densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from .vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features, vgg19_features, vgg19_bn_features


_base_to_features = {
    'resnet18': resnet18_features,
    'resnet34': resnet34_features,
    'resnet50': resnet50_features,
    'resnet101': resnet101_features,
    'resnet152': resnet152_features,
    'resnet50_inat': resnet50_features_inat,
    'densenet121': densenet121_features,
    'densenet161': densenet161_features,
    'densenet169': densenet169_features,
    'densenet201': densenet201_features,
    'vgg11': vgg11_features,
    'vgg11_bn': vgg11_bn_features,
    'vgg13': vgg13_features,
    'vgg13_bn': vgg13_bn_features,
    'vgg16': vgg16_features,
    'vgg16_bn': vgg16_bn_features,
    'vgg19': vgg19_features,
    'vgg19_bn': vgg19_bn_features
}


def init_backbone(backbone):

    backbone_name = backbone.name
    use_pretrained = backbone.pretrained

    features = _base_to_features[backbone_name](pretrained=use_pretrained)

    return features
