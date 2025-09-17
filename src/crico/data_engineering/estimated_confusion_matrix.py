import argparse
import math
from itertools import dropwhile, groupby, takewhile

parser = argparse.ArgumentParser(description="")

parser.add_argument("--gold_counts", type=str)
parser.add_argument("--dictionary_counts", type=str)


def get_count_dictionary(counts_path: str) -> dict[str, int]:
    with open(counts_path) as f:
        instances = f.readlines()

    def parse(inst: str) -> tuple[str, int]:
        # raw_term, raw_count = inst.split()
        # return raw_term.strip(), int(raw_count.strip())
        # remember kids, use TSVs or JSON instead of pretty
        # data vis or else you might end up doing insane things like this
        inst_ls = list(inst.strip())
        # yeah let's edit the list in place and not just
        # return a view of the data structure like rational
        # adults (serves me right for not doing that myself
        # via the stride interface)
        inst_ls.reverse()
        inst_iter = iter(inst_ls)
        reversed_count_ls = list(takewhile(str.isnumeric, inst_iter))
        reversed_count_ls.reverse()
        reversed_term_ls = list(dropwhile(str.isspace, inst_iter))
        reversed_term_ls.reverse()
        # print(inst, reversed_count_ls, reversed_term_ls)
        return "".join(reversed_term_ls), int("".join(reversed_count_ls))

    return {term: count for term, count in map(parse, instances)}


def main() -> None:
    args = parser.parse_args()
    gold_counts = get_count_dictionary(args.gold_counts)
    dictionary_counts = get_count_dictionary(args.dictionary_counts)

    def difference(term: str) -> int:
        num_gold_occurences = gold_counts.get(term, 0)
        num_dictionary_occurences = dictionary_counts.get(term, 0)
        return num_dictionary_occurences - num_gold_occurences

    def agreement(term_tup) -> int:
        term, _ = term_tup
        term_difference = difference(term)
        if term_difference == 0:
            return 0
        return math.copysign(1, term_difference)

    def to_agreement_tuple(diff_sign, diff_iter) -> tuple[int, list[str]]:
        return diff_sign, [term for term, _ in diff_iter]

    sorted_by_difference = sorted(dictionary_counts.items(), key=agreement)
    agreement_tuples = [
        to_agreement_tuple(diff_sign, counts_iter)
        for diff_sign, counts_iter in groupby(sorted_by_difference, key=agreement)
    ]

    for diff_sign, terms in agreement_tuples:
        match diff_sign:
            case -1:
                print("NOT ALL GOLD OCCURRENCES OF TERM FOUND BY DICTIONARY")
            case 0:
                print(
                    "TERMS WITH EQUAL NUMBER OF OCCURRENCES IN GOLD AND IN DICTIONARY"
                )
            case 1:
                print("NOT ALL TERM OCCURRENCES FOUND BY DICTIONARY IN GOLD")
        for term in terms:
            print(
                f"{term}: gold count {gold_counts.get(term, 0)} dictionary count {dictionary_counts.get(term, 0)}"
            )


if __name__ == "__main__":
    main()
