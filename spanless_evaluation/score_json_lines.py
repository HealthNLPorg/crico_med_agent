import logging
import argparse
import json
from collections.abc import Iterable
from operator import itemgetter
from itertools import groupby


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
med_dict = dict[str, str | int | bool]


def __load_med_dicts(jsonl_path: str) -> Iterable[med_dict]:
    with open(jsonl_path, mode="rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def __shared_med_dicts(
    d_ls_1: list[med_dict], d_ls_2: list[med_dict], meds: set[str]
) -> Iterable[tuple[med_dict, med_dict]]:
    for med in meds:
        d_ls_1_matches = [
            med_dict for med_dict in d_ls_1 if med_dict["medication"] == med
        ]
        d_ls_2_matches = [
            med_dict for med_dict in d_ls_2 if med_dict["medication"] == med
        ]
        assert len(d_ls_1_matches) == len(d_ls_2_matches) == 1
        for d_1, d_2 in zip(d_ls_1_matches, d_ls_2_matches):
            yield d_1, d_2


# def __attr_confusion_mattrix(
#     ground_med_dicts: list[med_dict],
#     pred_med_dicts: list[med_dict],
#     true_positive_meds: set[str],
#     false_positive_meds: set[str],
#     false_negative_meds: set[str],
#     attr_key: str,
# ) -> tuple[int, int, int]:
#     attr_matches_true_positive = {
#         ground_med_dict[attr_key] and pred_med_dict[attr_key]
#         for ground_med_dict, pred_med_dict in __shared_med_dicts(
#             ground_med_dicts, pred_med_dicts, true_positive_meds
#         )
#     }
#     total_true_positive_attr = len(
#         {has_inst for has_inst in attr_matches_true_positive if has_inst}
#     )
#     tp_med_fp_attr = len(
#         {has_inst for has_inst in attr_matches_true_positive if not has_inst}
#     )
#     fp_med_fp_attr = len(

#     )


def __study_id_confusion_matrix(
    ground_med_dicts: list[med_dict],
    pred_med_dicts: list[med_dict],
    # convention is TP, FP, FN
) -> dict[str, tuple[int, int, int]]:
    # Contract from upstream is
    # meds are grouped into equivalence classes
    # e.g. if there are multiple med mentions in a study which
    # stripped and lowercased are the same then
    # they're the same, and instructions/conditions percolate up,
    # TLDR using set arithmetic here should suffice

    ground_meds = {med_dict["medication"] for med_dict in ground_med_dicts}
    pred_meds = {med_dict["medication"] for med_dict in pred_med_dicts}
    true_positive_meds = ground_meds.intersection(pred_meds)
    false_positive_meds = pred_meds.difference(ground_meds)
    false_negative_meds = ground_meds.difference(pred_meds)

    return {
        "medication": (
            len(true_positive_meds),
            len(false_positive_meds),
            len(false_positive_meds),
        )
    }


# Schema is:
# {
#     "study_id": ... ,
#     "medication": ... ,
#     "has_at_least_one_instruction": ... ,
#     "has_at_least_one_condition": ... ,
# }
def __evaluate(
    ground_truth_jsonl: str,
    prediction_jsonl: str,
) -> None:
    # Doing micro-averaged/naive approach to start
    ground_truth_med_dicts = sorted(
        __load_med_dicts(ground_truth_jsonl), key=itemgetter("study_id")
    )
    prediction_med_dicts = sorted(
        __load_med_dicts(prediction_jsonl), key=itemgetter("study_id")
    )
    ground_truth_study_id_to_meds = {
        study_id: list(med_dicts)
        for study_id, med_dicts in groupby(
            ground_truth_med_dicts, key=itemgetter("study_id")
        )
    }

    prediction_study_id_to_meds = {
        study_id: list(med_dicts)
        for study_id, med_dicts in groupby(
            prediction_med_dicts, key=itemgetter("study_id")
        )
    }


def main() -> None:
    args = parser.parse_args()
    __evaluate(
        args.ground_truth_jsonl,
        args.prediction_jsonl,
    )


if __name__ == "__main__":
    main()
