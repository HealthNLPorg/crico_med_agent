import logging
import argparse


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--prediction_jsonl",
    type=str,
    default=None,
)
parser.add_argument(
    "--ground_truth_jsonl",
    type=str,
    default=None,
)


def evaluate(
    ground_truth_jsonl: str,
    prediction_jsonl: str,
) -> None:
    pass


def main() -> None:
    args = parser.parse_args()
    evaluate(
        args.ground_truth_jsonl,
        args.prediction_jsonl,
    )


if __name__ == "__main__":
    main()
