"""Bounded offline CLI tests: python -B -m unittest test_cli -v."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.cli import main


class CLITests(unittest.TestCase):
    def error(self, args, message):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as result:
            main(args)
        self.assertEqual(result.exception.code, 2)
        self.assertIn(message, stderr.getvalue())

    def test_help(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            main(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_unimplemented_tasks(self):
        for task in ("caption", "colorization"):
            for command in ("train", "predict"):
                self.error([command, "--task", task], "not implemented yet")

    def test_invalid_prediction_arguments(self):
        base = ["predict", "--task", "classification"]
        self.error(base, "requires --image and --checkpoint")
        self.error(base + ["--image", "x", "--checkpoint", "y", "--top-k", "0"], "must be positive")
        self.error(base + ["--unknown"], "unrecognized arguments")
        self.error(["train", "--task", "classification", "--config", "does-not-exist.yaml"], "Configuration not found")
        self.error(base + ["--image", "does-not-exist.jpg", "--checkpoint", "missing.pth"], "Image not found")
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "image.png"
            image.write_bytes(b"not an image")
            self.error(base + ["--image", str(image), "--checkpoint", str(Path(tmp) / "missing.pth")], "Checkpoint not found")

    def test_training_delegation_and_argv_restoration(self):
        original = sys.argv
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.yaml"
            config.write_text("epochs: 1\n")
            for kind, module in (("scratch", "src.training.train"), ("transfer", "src.training.train_transfer")):
                observed = []
                def capture(name, run_name):
                    observed.append((name, run_name, sys.argv[:]))
                with patch("src.cli.runpy.run_module", side_effect=capture):
                    main(["train", "--task", "classification", "--model", kind,
                          "--config", str(config), "--epochs", "1", "--batch-size", "2"])
                self.assertEqual(observed, [(module, "__main__", [module, "--config", str(config), "--epochs", "1", "--batch-size", "2"])])
                self.assertIs(sys.argv, original)

    def test_json_output_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prediction.json"
            args = ["predict", "--task", "classification", "--image", "bird.jpg",
                    "--checkpoint", "model.pth", "--output", str(output)]
            result = {"predictions": [{"class_id": 0, "probability": 0.8}]}
            with patch("src.inference.classification.predict", return_value=result), contextlib.redirect_stdout(io.StringIO()) as stdout:
                main(args)
            self.assertEqual(json.loads(stdout.getvalue()), result)
            self.assertEqual(json.loads(output.read_text()), result)
            self.error(args, "Output already exists")
            self.assertEqual(json.loads(output.read_text()), result)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable")
class CheckpointTests(unittest.TestCase):
    def test_scratch_checkpoint_formats(self):
        import torch
        from src.models.scratch_cnn import ScratchCNN
        from src.inference.classification import load_classifier
        torch.set_num_threads(2)
        model = ScratchCNN(2).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pth"
            for names in (None, ["bird A", "bird B"]):
                state = {"model_state_dict": model.state_dict()}
                if names is not None:
                    state["class_names"] = names
                torch.save(state, path)
                loaded, labels, mean, std, kind = load_classifier(tmp)
                self.assertEqual(labels, names)
                self.assertEqual(mean, (0.5,) * 3)
                self.assertEqual(std, mean)
                self.assertEqual(kind, "scratch")
                self.assertFalse(loaded.training)
                self.assertTrue(all(torch.equal(v, loaded.state_dict()[k]) for k, v in model.state_dict().items()))
            label_path = Path(tmp) / "labels.json"
            label_path.write_text('["only one"]')
            with self.assertRaisesRegex(ValueError, "class count"):
                load_classifier(path, class_names_path=label_path)

    @unittest.skipUnless(importlib.util.find_spec("torchvision"), "torchvision unavailable: real image inference requires it")
    def test_real_image_inference_both_models(self):
        import torch
        from PIL import Image
        from src.models.scratch_cnn import ScratchCNN
        from src.models.transfer_cnn import TransferCNN, save_model_config
        from src.inference.classification import predict
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "gray.png"
            Image.new("L", (40, 70), 100).save(image)
            for kind in ("scratch", "transfer"):
                run = root / kind
                run.mkdir()
                if kind == "scratch":
                    model = ScratchCNN(2)
                else:
                    model = TransferCNN(2, backbone="resnet18", pretrained=False, strategy="frozen")
                    save_model_config(run / "model_config.json", model, ["a", "b"])
                checkpoint = run / "custom.pth"
                torch.save({"model_state_dict": model.state_dict(), "class_names": ["a", "b"]}, checkpoint)
                with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("Inference must not download weights")):
                    result = predict(image, checkpoint, device="cpu", top_k=5)
                    again = predict(image, checkpoint, device="cpu", top_k=5)
                self.assertEqual(result, again)
                self.assertEqual(result["model"], kind)
                self.assertEqual(len(result["predictions"]), 2)
                self.assertAlmostEqual(sum(p["probability"] for p in result["predictions"]), 1, places=5)
                self.assertGreaterEqual(result["predictions"][0]["probability"], result["predictions"][1]["probability"])


if __name__ == "__main__":
    unittest.main()
