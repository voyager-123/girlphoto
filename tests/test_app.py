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
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from app import Store, make_server


def picture(color):
    stream = io.BytesIO()
    Image.new("RGB", (60, 90), color).save(stream, format="PNG")
    return stream.getvalue()


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.a = self.store.import_photo(picture("red"), "001__warm.png", cohort="person1")["id"]
        self.b = self.store.import_photo(picture("green"), "001__cool.png", cohort="person1")["id"]
        self.c = self.store.import_photo(picture("blue"), "002__cool.png", cohort="person1")["id"]

    def tearDown(self):
        self.temp.cleanup()

    def label(self, photo, **extra):
        payload = {"photo_id": photo, "reviewer": "alice", "quality": 2, "styles": {"retro": "yes", "fresh": "unknown"}}
        payload.update(extra)
        self.store.label(payload)

    def vote(self, **extra):
        payload = {"reviewer": "alice", "left": self.a, "right": self.b, "style_id": "retro", "outcome": "left"}
        payload.update(extra)
        self.store.save_pair(payload)

    def export_json(self):
        with zipfile.ZipFile(io.BytesIO(self.store.export())) as z:
            return json.loads(z.read("dataset.json"))

    def test_import_dedup_and_group_split(self):
        self.assertTrue(self.store.import_photo(picture("red"), "different.png")["duplicate"])
        photos = self.store.state("alice")["photos"]
        self.assertEqual(len(photos), 3)
        self.assertEqual(len({p["split"] for p in photos}), 1)
        self.assertEqual(len({p["group_id"] for p in photos}), 2)
        with self.assertRaises(ValueError):
            self.store.import_photo(b"not an image", "bad.jpg")

    def test_missing_and_unknown_stay_distinct(self):
        self.label(self.a, quality=None)
        data = self.export_json()
        label = data["labels"][0]
        self.assertEqual(label["styles"]["fresh"], "unknown")
        self.assertNotIn("cinematic", label["styles"])
        self.assertIsNone(label["quality"])
        with self.assertRaises(ValueError):
            self.label(self.b, quality=True)

    def test_reviewer_isolation_and_persistence(self):
        self.label(self.a)
        again = Store(self.temp.name)
        self.assertIsNotNone(next(p for p in again.state("alice")["photos"] if p["id"] == self.a)["label"])
        self.assertTrue(all(p["label"] is None for p in again.state("bob")["photos"]))

    def test_pair_requires_same_original_and_confirmed_style(self):
        with self.assertRaises(ValueError):
            self.vote()
        for photo in (self.a, self.b, self.c):
            self.label(photo)
        result = self.store.pair_next("alice", "retro")
        self.assertEqual(result["remaining"], 1)
        self.assertEqual({p["id"] for p in result["pair"]}, {self.a, self.b})
        with self.assertRaises(ValueError):
            self.vote(right=self.c)

    def test_pair_order_normalization_and_skip(self):
        self.label(self.a)
        self.label(self.b)
        self.vote(left=self.b, right=self.a, outcome="left")
        pair = self.export_json()["pairs"][0]
        self.assertEqual(pair[pair["outcome"]], self.b)
        self.assertTrue(pair["eligible"])
        self.assertEqual(self.store.pair_next("alice", "retro")["remaining"], 0)
        self.vote(outcome="skip")
        self.assertFalse(self.export_json()["pairs"][0]["eligible"])
        self.assertEqual(self.store.pair_next("alice", "retro", True)["remaining"], 1)

    def test_undo_restores_previous_label_and_pair(self):
        self.label(self.a, quality=1)
        self.label(self.a, quality=3)
        self.store.undo("alice")
        self.assertEqual(self.export_json()["labels"][0]["quality"], 1)
        self.label(self.b)
        self.vote(outcome="tie")
        self.vote(outcome="right")
        self.store.undo("alice")
        self.assertEqual(self.export_json()["pairs"][0]["outcome"], "tie")

    def test_updated_style_invalidates_old_pair_without_erasing_it(self):
        self.label(self.a)
        self.label(self.b)
        self.vote()
        self.label(self.a, styles={"retro": "no"})
        self.assertFalse(self.export_json()["pairs"][0]["eligible"])
        self.store.undo("alice")
        self.assertTrue(self.export_json()["pairs"][0]["eligible"])

    def test_group_move_restricted_and_cohort_split_shared(self):
        self.store.update_split({"cohort_id": "person1", "split": "test"})
        self.assertEqual({p["split"] for p in self.store.state("alice")["photos"]}, {"test"})
        self.label(self.a)
        self.label(self.b)
        self.vote()
        with self.assertRaises(ValueError):
            self.store.update_group({"photo_id": self.a, "group_id": "new", "cohort_id": "new", "split": "train"})

    def test_export_formula_safety_and_new_style_masks(self):
        self.label(self.a, note="=2+3")
        self.store.add_style("柔和", "用于测试的柔和风格")
        with zipfile.ZipFile(io.BytesIO(self.store.export())) as z:
            self.assertIn("'=2+3", z.read("quality.csv").decode("utf-8-sig"))
            self.assertIn("unreviewed", z.read("styles.csv").decode("utf-8-sig"))
            self.assertEqual(json.loads(z.read("dataset.json"))["labels"][0]["note"], "=2+3")


class HttpTests(unittest.TestCase):
    def test_http_flow_and_security(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            server = make_server(store, 0)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/state") as response:
                    token = json.load(response)["token"]
                payload = json.dumps({"name": "x__v1.png", "data": base64.b64encode(picture("orange")).decode()}).encode()
                with self.assertRaises(HTTPError) as missing:
                    urlopen(Request(base + "/api/import", payload, headers={"Content-Type": "application/json"}))
                self.assertEqual(missing.exception.code, 403)
                with urlopen(Request(base + "/api/import", payload, headers={"Content-Type": "application/json", "X-Girlphoto-Token": token})) as response:
                    photo_id = json.load(response)["id"]
                with urlopen(base + "/media/" + photo_id) as response:
                    self.assertEqual(response.headers["Content-Type"], "image/jpeg")
                    self.assertGreater(len(response.read()), 100)
                with urlopen(base + "/api/export") as response:
                    self.assertTrue(zipfile.is_zipfile(io.BytesIO(response.read())))
                with self.assertRaises(HTTPError) as missing:
                    urlopen(base + "/media/../../app.py")
                self.assertEqual(missing.exception.code, 404)
                with self.assertRaises(HTTPError) as wrong_host:
                    urlopen(Request(base + "/", headers={"Host": "evil.example"}))
                self.assertEqual(wrong_host.exception.code, 403)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == "__main__":
    unittest.main()
