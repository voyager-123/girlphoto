import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from girlphoto_ml.__main__ import predict
from girlphoto_ml.data import Manifest, read_image
from girlphoto_ml.demo import make_demo
from girlphoto_ml.models import Adjuster, Scorer, load_checkpoint, save_checkpoint
from girlphoto_ml.renderer import render
from girlphoto_ml.train import sample_loss, train_adjuster, train_scorer


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_identity_bounds_and_renderer_gradients(self):
        x = torch.rand(2, 3, 24, 32) * .4 + .3
        self.assertTrue(torch.allclose(render(x, torch.zeros(2, 6)), x, atol=1e-7))
        p = torch.full((2, 6), .1, requires_grad=True)
        result = render(x, p)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0)
        self.assertGreaterEqual(float(result.detach().min()), 0)
        self.assertLessEqual(float(result.detach().max()), 1)

    def test_model2_zero_strength_and_parameter_bounds(self):
        model = Adjuster(2)
        with torch.no_grad():
            model.head[-1].bias.fill_(2)
        source = torch.rand(2, 3, 32, 32)
        p = model(source, torch.tensor([0, 1]), torch.tensor([0., .4]))
        self.assertTrue(torch.equal(p[0], torch.zeros(6)))
        self.assertLessEqual(float(p[1].detach().abs().max()), .4)
        self.assertTrue(torch.allclose(render(source, p)[0], source[0], atol=1e-7))

    def test_unknown_styles_and_absent_strength_have_no_loss(self):
        model = Scorer(2)
        x = torch.rand(1, 3, 32, 32)
        self.assertIsNone(sample_loss(model, x, [{"styles": {"retro": "unknown"}}], ["retro", "fresh"]))
        loss = sample_loss(model, x, [{"styles": {"retro": "yes"}}], ["retro", "fresh"])
        loss.backward()
        self.assertEqual(float(model.style.weight.grad[1].abs().sum()), 0)
        self.assertIsNone(model.strength.weight.grad)
        self.assertIsNone(model.quality.weight.grad)

    def test_frozen_scorer_still_propagates_to_parameters(self):
        scorer = Scorer(1).eval()
        for item in scorer.parameters():
            item.requires_grad_(False)
        source = torch.rand(1, 3, 32, 32) * .4 + .3
        params = torch.full((1, 6), .05, requires_grad=True)
        score = scorer.rank(source, render(source, params), torch.tensor([0]))
        score.sum().backward()
        self.assertGreater(float(params.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in scorer.parameters()))

    def test_manifest_blocks_split_leakage_and_invalid_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = make_demo(folder)
            base = json.loads(path.read_text())
            invalid = copy.deepcopy(base)
            invalid["samples"][1]["split"] = "test"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "Split leakage"):
                Manifest(path)
            invalid = copy.deepcopy(base)
            invalid["tasks"][0]["parameters"][0] = 1.1
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "normalized values"):
                Manifest(path)
            invalid = copy.deepcopy(base)
            invalid["pairs"][0]["source"] = "g1_original"
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "share original group"):
                Manifest(path)

    def test_checkpoint_roundtrip_and_kind_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"model.pt"
            model = Scorer(1).eval()
            save_checkpoint(path, model, "scorer", ["retro"], 32, {}, True)
            restored, bundle = load_checkpoint(path, "scorer")
            image = torch.rand(1, 3, 32, 32)
            self.assertTrue(torch.equal(model(image)["style_logits"], restored(image)["style_logits"]))
            self.assertTrue(bundle["demo_only"])
            with self.assertRaises(ValueError):
                load_checkpoint(path, "adjuster")

    def test_training_both_models_and_full_size_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = make_demo(root)
            first = train_scorer(manifest, root/"m1.pt", steps=2, batch_size=2, size=32)
            self.assertGreater(first["labeled_samples"], 0)
            second = train_adjuster(manifest, root/"m2.pt", scorer_path=root/"m1.pt", steps=2, batch_size=2, size=32)
            self.assertTrue(second["demo_only"])
            self.assertEqual(second["validation_tasks"], 2)
            source = root/"images/g4_original.png"
            result = predict(root/"m2.pt", source, "retro", 0, root/"result.png")
            self.assertTrue(result["demo_only"])
            self.assertTrue(torch.equal(read_image(source), read_image(root/"result.png")))
            exported = json.loads((root/"result.parameters.json").read_text())
            self.assertTrue(all(value == 0 for value in exported["normalized"].values()))
            with self.assertRaises(ValueError):
                predict(root/"m2.pt", source, "retro", .5, source)


if __name__ == "__main__":
    unittest.main()
