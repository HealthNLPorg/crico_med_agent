import argparse
import os
import re
import json
import logging
from ast import literal_eval
from itertools import chain, islice
from typing import cast, Any
from collections.abc import Iterable
from collections import Counter
from operator import itemgetter
from functools import partial

import numpy as np
import pandas as pd
from anafora_data import (
    AnaforaDocument,
    Instruction,
    InstructionCondition,
    Dosage,
    Frequency,
    Medication,
    MedicationAttribute,
)
from utils import basename_no_ext, mkdir

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--paired_anafora_dir",
    type=str,
    default=None,
)

parser.add_argument(
    "--input_tsv",
    type=str,
    default=None,
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
)


def anafora_process(
    anafora_dir: str,
    output_dir: str,
) -> None:
    pass


def tsv_process(
    input_tsv: str,
    output_dir: str,
) -> None:
    pass


def main() -> None:
    args = parser.parse_args()
    if args.paired_anafora_dir is not None:
        anafora_process(args.paired_anafora_dir, args.output_dir)
    elif args.input_tsv is not None:
        tsv_process(args.input_tsv, args.output_dir)


if __name__ == "__main__":
    main()
