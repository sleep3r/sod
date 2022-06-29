from addict import Dict


def get_config(arch: int):
    cfg = Dict(
        arch=arch,
        channels=[24, 40, 112, 320],
        RFB_aggregated_channel=[32, 64, 128],
        frequency_radius=16,
        denoise=0.93,
        gamma=0.1,
        img_size=320,
    )

    img_sizes = {
        0: 320,
        1: 320,
        2: 352,
        3: 384,
        4: 448,
        5: 512,
        6: 576,
        7: 640,
    }

    cfg.img_size = img_sizes[cfg.arch]
    return cfg
