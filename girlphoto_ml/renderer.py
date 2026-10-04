"""A small, differentiable, versioned sRGB adjustment pipeline."""
import torch

ENGINE = "girlphoto-global-srgb-v1"
PARAMETERS = ("exposure", "contrast", "warmth", "tint", "saturation", "shadows")


def render(image, parameters):
    """BCHW float RGB [0,1], Bx6 normalized parameters [-1,1]. Zero is identity."""
    if image.ndim != 4 or image.shape[1] != 3 or parameters.shape != (image.shape[0], 6):
        raise ValueError("Expected BCHW RGB images and Bx6 parameters.")
    p = parameters[:, :, None, None]
    x = image * torch.pow(2.0, p[:, 0:1] * 2.0)
    x = (x - 0.5) * (1.0 + p[:, 1:2] * 0.5) + 0.5
    warmth, tint = p[:, 2:3] * 0.15, p[:, 3:4] * 0.15
    x = x + torch.cat((warmth - tint / 2, tint, -warmth - tint / 2), dim=1)
    luminance = (x * x.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]).sum(1, keepdim=True)
    x = luminance + (x - luminance) * (1.0 + p[:, 4:5] * 0.8)
    x = x + p[:, 5:6] * 0.15 * (1.0 - luminance.clamp(0, 1)) ** 2
    return x.clamp(0, 1)


def describe(parameters):
    p = [float(v) for v in parameters]
    return {"engine": ENGINE, "normalized": dict(zip(PARAMETERS, p)),
            "effective": {"exposure_stops": 2*p[0], "contrast_multiplier": 1+0.5*p[1],
                          "warmth_rgb_offset": 0.15*p[2], "tint_rgb_offset": 0.15*p[3],
                          "saturation_multiplier": 1+0.8*p[4], "shadow_offset": 0.15*p[5]},
            "note": "Parameters apply only to this engine; they are not Lightroom slider values."}
