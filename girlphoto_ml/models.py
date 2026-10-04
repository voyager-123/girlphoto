import torch
from torch import nn

from .renderer import ENGINE, PARAMETERS


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        incoming = 3
        for outgoing in (16, 32, 64):
            blocks += [nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1),
                       nn.GroupNorm(4, outgoing), nn.SiLU()]
            incoming = outgoing
        self.layers = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, 64), nn.SiLU())

    def forward(self, image):
        return self.layers(image)


class Scorer(nn.Module):
    """Model 1: separate style, quality, intensity and source-conditioned preference heads."""
    def __init__(self, num_styles):
        super().__init__()
        self.encoder = Encoder()
        self.style = nn.Linear(64, num_styles)
        self.quality = nn.Linear(64, 3)
        self.strength = nn.Linear(64, num_styles)
        self.style_embedding = nn.Embedding(num_styles, 8)
        self.preference = nn.Sequential(nn.Linear(64*3+8, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, image):
        features = self.encoder(image)
        return {"style_logits": self.style(features), "quality_logits": self.quality(features),
                "strength": self.strength(features).sigmoid()}

    def rank(self, source, image, style_id):
        original, edited = self.encoder(source), self.encoder(image)
        inputs = torch.cat((original, edited, edited-original, self.style_embedding(style_id)), dim=1)
        return self.preference(inputs).squeeze(1)


class Adjuster(nn.Module):
    """Model 2: original + style + strength -> six bounded adjustment parameters."""
    def __init__(self, num_styles):
        super().__init__()
        self.encoder = Encoder()
        self.style_embedding = nn.Embedding(num_styles, 8)
        self.head = nn.Sequential(nn.Linear(73, 64), nn.SiLU(), nn.Linear(64, len(PARAMETERS)))
        # Begin at the identity instead of applying a random severe retouch.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, original, style_id, strength):
        features = torch.cat((self.encoder(original), self.style_embedding(style_id), strength[:, None]), dim=1)
        return self.head(features).tanh() * strength[:, None]


def save_checkpoint(path, model, kind, styles, image_size, supervision, demo=False):
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "kind": kind, "engine": ENGINE, "styles": styles,
                "image_size": image_size, "supervision": supervision, "demo_only": bool(demo),
                "state_dict": model.state_dict()}, path)


def load_checkpoint(path, kind, device="cpu"):
    bundle = torch.load(path, map_location=device, weights_only=True)
    if bundle.get("format_version") != 1 or bundle.get("kind") != kind or bundle.get("engine") != ENGINE:
        raise ValueError("Checkpoint kind/version/renderer does not match this program.")
    model = (Scorer if kind == "scorer" else Adjuster)(len(bundle["styles"])).to(device)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()
    return model, bundle
