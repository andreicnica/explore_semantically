from . import resnet_unet


def get_model(cfg, **kwargs):
    assert hasattr(cfg, "name") and cfg.name in MODELS,\
        f"Please provide a valid model name. {cfg}"
    return MODELS[cfg.name](cfg, **kwargs)


MODELS = {
    "ResNetUNet": resnet_unet.ResNetUNet,
}




