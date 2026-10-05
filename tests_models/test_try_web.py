import base64
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
import torch

from app import Store, make_server
from girlphoto_ml.models import Adjuster, Scorer, save_checkpoint
from try_service import TryService


def photo(size=(90, 120)):
    stream = io.BytesIO()
    Image.new("RGB", size, (120, 100, 80)).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()


class TryWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "models"
        self.model_dir.mkdir()
        self.checkpoint = self.model_dir / "model2.pt"
        model = Adjuster(2)
        with torch.no_grad():
            model.head[-1].bias.fill_(.4)
        save_checkpoint(self.checkpoint, model, "adjuster", ["retro", "fresh"], 32,
                        {"trained_styles": ["retro"], "uses_scorer": True}, True)
        self.service = TryService(self.model_dir)
        self.catalog = self.service.catalog()
        self.payload = {"model_id": self.catalog["models"][0]["id"], "style": "retro", "strength": .7, "image": photo()}

    def tearDown(self):
        self.temp.cleanup()

    def test_catalog_filters_scorers_bad_weights_and_untrained_styles(self):
        save_checkpoint(self.model_dir/"model1.pt", Scorer(2), "scorer", ["retro", "fresh"], 32, {}, True)
        (self.model_dir/"broken.pt").write_bytes(b"broken")
        catalog = self.service.catalog()
        self.assertEqual(len(catalog["models"]), 1)
        self.assertEqual(catalog["models"][0]["styles"], [{"id": "retro", "name": "复古"}])
        self.assertEqual(catalog["skipped"], 2)
        self.assertTrue(catalog["models"][0]["demo_only"])

    def test_zero_strength_identity_and_nonzero_output(self):
        output = self.service.predict({**self.payload, "strength": 0})
        self.assertEqual(output["before_png"], output["after_png"])
        self.assertEqual(set(output["parameters"]["normalized"].values()), {0})
        self.assertEqual(output["parameters"]["source_size"], [90, 120])
        output = self.service.predict(self.payload)
        self.assertNotEqual(output["before_png"], output["after_png"])
        self.assertEqual(len(output["parameters"]["checkpoint_sha256"]), 64)
        self.assertTrue(output["parameters"]["demo_only"])
        # No uploaded image or inference output is written to disk.
        self.assertEqual([p.name for p in self.model_dir.iterdir()], ["model2.pt"])

    def test_preview_resize_retains_parameters_from_original(self):
        output = self.service.predict({**self.payload, "image": photo((1800, 900))})
        self.assertEqual(output["parameters"]["source_size"], [1800, 900])
        self.assertEqual(output["parameters"]["preview_size"], [1600, 800])
        with Image.open(io.BytesIO(base64.b64decode(output["after_png"]))) as image:
            self.assertEqual(image.size, (1600, 800))

    def test_invalid_requests_and_model_change(self):
        for changes in ({"strength": True}, {"strength": float("nan")}, {"strength": 2},
                        {"model_id": "../../other.pt"}, {"style": "fresh"}, {"image": "broken"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.service.predict({**self.payload, **changes})
        self.checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "模型列表已变化"):
            self.service.predict(self.payload)

    def test_http_page_upload_token_and_training_store_unchanged(self):
        store = Store(self.root/"data")
        server = make_server(store, 0, self.model_dir)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base+"/try") as response:
                self.assertIn('生成调色结果', response.read().decode())
            with urlopen(base+"/api/try/models") as response:
                catalog = json.load(response)
            body = json.dumps({**self.payload, "model_id": catalog["models"][0]["id"]}).encode()
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base+"/api/try/predict", body))
            self.assertEqual(error.exception.code, 403)
            with urlopen(Request(base+"/api/try/predict", body, headers={"Content-Type": "application/json", "X-Girlphoto-Token": catalog["token"]})) as response:
                self.assertIn("after_png", json.load(response))
            self.assertEqual(store.state("我")["photos"], [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
