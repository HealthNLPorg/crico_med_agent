import argparse
import json
import logging
from collections.abc import Iterable
from itertools import groupby
from operator import itemgetter
from typing import cast

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
MedDict = dict[str, str | int | bool]


def __precision(tp: int, fp: int) -> float:
    return tp / (tp + fp)


def __recall(tp: int, fn: int) -> float:
    return tp / (tp + fn)


def __f1(precision: float, recall: float) -> float:
    return (2 * precision * recall) / (precision + recall)


def __scores(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = __precision(tp, fp)
    recall = __recall(tp, fn)
    return precision, recall, __f1(precision, recall)


def __get_type_level_scores(
    study_id_to_confusion_matrix: dict[str, dict[str, tuple[int, int, int]]],
    type_key: str,
) -> tuple[float, float, float]:
    total_tp, total_fp, total_fn = map(
        sum, zip(*map(itemgetter(type_key), study_id_to_confusion_matrix.values()))
    )
    return __scores(total_tp, total_fp, total_fn)


def __load_med_dicts(jsonl_path: str) -> Iterable[MedDict]:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def __med_matches(med_dicts: Iterable[MedDict], meds: set[str]) -> Iterable[MedDict]:
    for med_dict in med_dicts:
        if med_dict["medication"] in meds:
            yield med_dict


def __shared_med_dicts(
    d_ls_1: list[MedDict], d_ls_2: list[MedDict], meds: set[str]
) -> Iterable[tuple[MedDict, MedDict]]:
    for med in meds:
        d_ls_1_matches = list(__med_matches(d_ls_1, {med}))
        d_ls_2_matches = list(__med_matches(d_ls_2, {med}))
        assert len(d_ls_1_matches) == len(d_ls_2_matches) == 1
        yield from zip(d_ls_1_matches, d_ls_2_matches)


def __attr_confusion_mattrix(
    ground_med_dicts: list[MedDict],
    pred_med_dicts: list[MedDict],
    true_positive_meds: set[str],
    false_positive_meds: set[str],
    false_negative_meds: set[str],
    attr_key: str,
    # convention is TP, FP, FN
) -> tuple[int, int, int]:
    tp_med_tp_attr = sum(
        1
        for ground_med_dict, pred_med_dict in __shared_med_dicts(
            ground_med_dicts, pred_med_dicts, true_positive_meds
        )
        # Need them to both be true
        if ground_med_dict[attr_key] and pred_med_dict[attr_key]
    )
    tp_med_fp_attr = sum(
        1
        for ground_med_dict, pred_med_dict in __shared_med_dicts(
            ground_med_dicts, pred_med_dicts, true_positive_meds
        )
        # Need ground to be false and predicted to be true
        if not ground_med_dict[attr_key] and pred_med_dict[attr_key]
    )
    tp_med_fn_attr = sum(
        1
        for ground_med_dict, pred_med_dict in __shared_med_dicts(
            ground_med_dicts, pred_med_dicts, true_positive_meds
        )
        # Need ground to be true and predicted to be false
        if ground_med_dict[attr_key] and not pred_med_dict[attr_key]
    )
    # Definitionally no FN/FP meds with TP attributes since
    # we're ignoring spans and only considering attributes
    # insofar as they are linked to meds
    # Definitionally no FN med with FP attr
    fp_med_fp_attr = (
        sum(  # Logic is an attribute's false-positivity is inherited from the
            # false-positivity of the predicted medication with which it is associated
            1
            for has_attr in map(
                itemgetter(attr_key),
                __med_matches(pred_med_dicts, false_positive_meds),
            )
            if has_attr
        )
    )
    # Definitionally no FN med with FP attr
    fn_med_fn_attr = sum(  # Logic is an attribute's false-negativity is inherited from the
        # false-negativity of the (un-predicted) ground truth medication with which it is associated
        1
        for has_attr in map(
            itemgetter(attr_key), __med_matches(ground_med_dicts, false_negative_meds)
        )
        if has_attr
    )
    total_attr_tp = tp_med_tp_attr
    total_attr_fp = tp_med_fp_attr + fp_med_fp_attr
    total_attr_fn = tp_med_fn_attr + fn_med_fn_attr
    return total_attr_tp, total_attr_fp, total_attr_fn


def __study_id_confusion_matrix(
    ground_med_dicts: list[MedDict],
    pred_med_dicts: list[MedDict],
    # convention is TP, FP, FN
) -> dict[str, tuple[int, int, int]]:
    # Contract from upstream is
    # meds are grouped into equivalence classes
    # e.g. if there are multiple med mentions in a study which
    # stripped and lowercased are the same then
    # they're the same, and instructions/conditions percolate up,
    # TLDR using set arithmetic here should suffice

    # For mypy
    ground_meds = {cast(str, med_dict["medication"]) for med_dict in ground_med_dicts}
    pred_meds = {cast(str, med_dict["medication"]) for med_dict in pred_med_dicts}
    true_positive_meds = ground_meds.intersection(pred_meds)
    false_positive_meds = pred_meds.difference(ground_meds)
    false_negative_meds = ground_meds.difference(pred_meds)

    return {
        "medication": (
            len(true_positive_meds),
            len(false_positive_meds),
            len(false_positive_meds),
        ),
        "instruction": __attr_confusion_mattrix(
            ground_med_dicts=ground_med_dicts,
            pred_med_dicts=pred_med_dicts,
            true_positive_meds=true_positive_meds,
            false_positive_meds=false_positive_meds,
            false_negative_meds=false_negative_meds,
            attr_key="has_at_least_one_instruction",
        ),
        "condition": __attr_confusion_mattrix(
            ground_med_dicts=ground_med_dicts,
            pred_med_dicts=pred_med_dicts,
            true_positive_meds=true_positive_meds,
            false_positive_meds=false_positive_meds,
            false_negative_meds=false_negative_meds,
            attr_key="has_at_least_one_condition",
        ),
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
    study_id_to_confusion_matrix = {
        study_id: __study_id_confusion_matrix(
            ground_med_dicts=ground_truth_study_id_to_meds.get(study_id, []),
            pred_med_dicts=prediction_study_id_to_meds.get(study_id, []),
        )
        for study_id in {
            *prediction_study_id_to_meds,
            *ground_truth_study_id_to_meds,
        }
    }
    for type_key in ["medication", "instruction", "condition"]:
        precision, recall, f1 = __get_type_level_scores(
            study_id_to_confusion_matrix=study_id_to_confusion_matrix, type_key=type_key
        )
        print(f"Scores for {type_key}:")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1:        {f1:.2f}")


def main() -> None:
    args = parser.parse_args()
    __evaluate(
        args.ground_truth_jsonl,
        args.prediction_jsonl,
    )


if __name__ == "__main__":
    main()
