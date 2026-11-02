"""Project entry point: python -m src.cli {train,predict} --task classification."""

import argparse
import json
from pathlib import Path
import pickle
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bird project training and inference CLI.", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "predict"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--task", required=True, choices=("classification", "caption", "colorization"))
        if name == "train":
            sub.add_argument("--model", choices=("scratch", "transfer"), default="scratch")
            sub.add_argument("--config", type=Path)
            sub.epilog = "Additional arguments are forwarded to the existing model trainer (e.g. --epochs, --batch-size)."
        else:
            sub.add_argument("--image", type=Path)
            sub.add_argument("--checkpoint", type=Path, help="Checkpoint file or run directory containing best_model.pth.")
            sub.add_argument("--model", choices=("auto", "scratch", "transfer"), default="auto")
            sub.add_argument("--class-names", type=Path, help="Optional JSON list in training class-index order (scratch only).")
            sub.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            sub.add_argument("--top-k", type=int, default=5)
            sub.add_argument("--output", type=Path, help="Also write the prediction JSON to this new file.")

    args, extra = parser.parse_known_args(argv)
    if args.task != "classification":
        parser.error(f"{args.task} is not implemented yet: this repository has no {args.task} model/trainer. Caption cleaning is not a caption generator.")
    try:
        if args.command == "train":
            if args.config is not None and not args.config.is_file():
                raise ValueError(f"Configuration not found: {args.config}")
            module = "src.training.train" if args.model == "scratch" else "src.training.train_transfer"
            forwarded = (["--config", str(args.config)] if args.config else []) + extra
            previous = sys.argv
            try:
                sys.argv = [module, *forwarded]
                runpy.run_module(module, run_name="__main__")
            finally:
                sys.argv = previous
        else:
            if extra:
                parser.error("unrecognized arguments: " + " ".join(extra))
            if args.image is None or args.checkpoint is None:
                parser.error("classification prediction requires --image and --checkpoint")
            if args.top_k <= 0:
                parser.error("--top-k must be positive")
            if args.output is not None and args.output.exists():
                parser.error(f"Output already exists: {args.output}; choose a new path")
            from src.inference.classification import predict
            result = predict(args.image, args.checkpoint, args.model, args.device, args.top_k, args.class_names)
            payload = json.dumps(result, indent=2)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as file:
                    file.write(payload + "\n")
            print(payload)
    except ImportError as error:
        parser.error(f"Missing or incompatible dependency: {error}. See docs/cli.md for requirements.")
    except (ValueError, OSError, KeyError, RuntimeError, pickle.UnpicklingError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
