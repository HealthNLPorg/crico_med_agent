import argparse
import json
import logging
import os
import pathlib
import re
from itertools import chain
from time import time
from typing import Callable, Iterable, cast
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from utils import basename_no_ext, mkdir
from text_engineering import deserialize_whitespace, serialize_whitespace

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
    help="Check the `get_examples` method for the possible formats for now",
)
parser.add_argument(
    "--sample_document",
    type=str,
)
parser.add_argument(
    "--sample_answer",
    type=str,
)
parser.add_argument("--prompt_file", type=str)
parser.add_argument(
    "--model_path",
    type=str,
    default="/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
)

parser.add_argument(
    "--attn_implementation",
    type=str,
    default="spda",
    choices=["spda", "flash_attention_2"],
)

parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--fancy_output", action="store_true")
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])

parser.add_argument("--text_column", type=str, default="text")

parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument(
    "--query_files",
    nargs="+",
    default=[],
    help="TSVs for now",
)

parser.add_argument(
    "--query_dir",
    help="TSVs for now",
)
parser.add_argument("--output_dir", type=str)

parser.add_argument(
    "--keep_columns",
    nargs="*",
    default=[],
    help="Columns to keep in the final frame",
)

name2path = {
    "llama2": "/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
    "llama3": "/lab-share/CHIP-Savova-e2/Public/resources/Meta-Llama-3-8B-Instruct/",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-1.5B-Instruct",
}

# {role: {system|user|assistant}, content: ...}
Message = dict[str, str]

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def process(
    model_name: str,
    text_column: str,
    prompt_file: str,
    query_files: str,
    max_new_tokens: int,
    output_dir: str,
    model_path: str | None,
    examples_file: str | None,
    sample_document: str | None,
    sample_answer: str | None,
    query_dir: str | None,
    batch_size: int,
        keeper_columns: list[str],
) -> None:
    final_path = ""
    if model_name is not None:
        final_path = name2path[model_name]
    else:
        final_path = model_path
    logger.info(f"Loading tokenizer and model for model name {final_path}")
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit
    # )
    system_prompt = get_system_prompt(prompt_file)
    logger.info("Building dataset")
    query_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=query_files,  # check cnlpt to see how to do this with splits (
        #     [
        #         os.path.join(args.query_dir, fn)
        #         for fn in os.listdir(args.query_dir)
        #         if fn.endswith("tsv")
        #     ]
        #     if args.query_dir
        #     else args.query_files
        # ),
    )
    query_dataset = query_dataset["train"]

    def few_shot_with_examples(
        examples: Iterable[tuple[str, str]],
    ) -> Callable[[str, str], list[Message]]:
        def _few_shot_prompt(s, q):
            return few_shot_prompt(system_prompt=s, query=q, examples=examples)

        return _few_shot_prompt

    if examples_file is not None:
        examples = get_examples(examples_file)
        if len(examples) > 0:
            get_prompt = few_shot_with_examples(examples=examples)

        else:
            ValueError("Empty examples file")

            get_prompt = empty_prompt
    elif sample_document is not None and sample_answer is not None:
        example = get_document_level_example(sample_document, sample_answer)
        if all(len(ex) > 0 for ex in example):
            get_prompt = few_shot_with_examples(examples=(example,))
        else:
            ValueError("Empty sample document and/or empty sample answer")

            get_prompt = empty_prompt
    else:
        get_prompt = zero_shot_prompt
    start = time()
    # checkpoint = final_path
    # weights_location = snapshot_download(repo_id=checkpoint)
    # config = AutoConfig.from_pretrained(checkpoint)
    # with init_empty_weights():
    #     model = AutoModelForCausalLM.from_config(config)
    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=weights_location, device_map="auto", no_split_module_classes=['Block']
    # )
    model = AutoModelForCausalLM.from_pretrained(final_path)
    tokenizer = AutoTokenizer.from_pretrained(final_path)
    seqgen_pipe = pipeline(
        "text-generation",
        # model=final_path,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=max_new_tokens,
    )
    end = time()
    logger.info(f"Loading model took {end - start} seconds")
    out_dir = output_dir
    out_fn_stem = pathlib.Path(
        query_dir if query_dir else "_".join(basename_no_ext(fn) for fn in query_files)
    ).stem
    tsv_out_fn = f"{out_fn_stem}.tsv"
    tsv_out_path = os.path.join(out_dir, tsv_out_fn)
    mkdir(out_dir)

    def format_chat(sample: dict) -> dict:
        return {
            "text": seqgen_pipe.tokenizer.apply_chat_template(
                get_prompt(system_prompt, deserialize_whitespace(sample[text_column])),
                tokenize=False,
                add_generation_prompt=False,
                truncate=True,
                # max_length=8_000,
            )
        }

    def predict(batch):
        batch["output"] = seqgen_pipe(batch["text"])
        return batch

    def serialize_output(batch):
        batch["serialized_output"] = [
            serialize_whitespace(
                output["generated_text"].split("<|eot_id|>assistant")[-1]
            )
            for output in batch["output"]
        ]
        return batch

    query_dataset = (
        query_dataset.map(format_chat)
        .map(predict, batched=True, batch_size=batch_size)
        .map(serialize_output)
        # .map(parse_output)
        # .filter(non_empty_json)
        # .filter(medication_non_hallucinatory)
        # .map(insert_mentions)
        # .map(clean_section)
        # .remove_columns(["text", "output", "json_output", text_column, "section_identifier"])
    )
    query_dataset.remove_columns(["text", "output"])
    query_dataframe = query_dataset.to_pandas()
    if len(keeper_columns) > 0:
        query_dataframe = query_dataframe[keeper_columns]
        #     [
        #         text_column,
        #         "section_identifier",
        #         "filename",
        #         "section_offsets",
        #         "serialized_output",
        #     ]
        # ]
    query_dataframe.to_csv(tsv_out_path, sep="\t", index=False)




def parse_output(sample: dict) -> dict:
    model_answer = sample["output"][0]["generated_text"].split("assistant")[-1].strip()
    sample["json_output"] = json.dumps(try_json(model_answer))
    return sample


def try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


# def medication_non_hallucinatory(sample: dict, text_column: str) -> bool:
#     def normalize(text: str, delim: str = "") -> str:
#         return delim.join(text.lower().split())

#     raw_medication = json.loads(sample["json_output"]).get("medication")
#     try:
#         return raw_medication is not None and normalize(raw_medication) in normalize(
#             sample[text_column], " "
#         )
#     except Exception:
#         logger.warning(
#             f"Issue with JSON sample {sample['json_output']} compared against {sample['section_body']}"
#         )
#         return False


def non_empty_json(sample: dict) -> bool:
    return len(sample["json_output"]) > 0


def empty_prompt(system_prompt: str, query: str) -> list[Message]:
    return []


def structure_response(index: int, query: str, answer: str) -> str:
    return f"Query {index}:\n{query}\nAnswer:\n{answer}\n\n"


def clean_section(sample: dict) -> dict:
    sample["section"] = str.title(" ".join(sample["section_identifier"].split("_")[1:]))
    return sample


def insert_mentions(sample: dict) -> dict:
    mention_components = {"medication", "instructions", "conditions"}
    components_dict = json.loads(sample["json_output"])
    for mention_component in mention_components:
        sample[mention_component] = "".join(
            components_dict.get(mention_component, "__UNK__")
        )
    return sample


def get_system_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, mode="rt", encoding="utf-8") as f:
        raw_prompt = f.read()
    return raw_prompt


def get_query_dataset(queries_file_path: str) -> Dataset:
    # NB, this will retrieve the extension with the "." at the front
    # e.g. ".txt" rather than "txt"
    suffix = pathlib.Path(queries_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(queries_file_path, sep="\t")
            raw_queries = cast(
                Iterable[str],
                (
                    full_dataframe["query"]
                    if "query" in full_dataframe.columns
                    else full_dataframe["sentence"]
                ),
            )

            def with_whitespace() -> Iterable[dict[str, str]]:
                for query in raw_queries:
                    yield {"text": deserialize_whitespace(query)}

            queries = Dataset.from_generator(with_whitespace)
        case ".txt" | "":
            with open(queries_file_path, mode="rt") as qf:
                query = qf.read()
            queries = Dataset.from_list([{"text": query}])
        case _:
            ValueError(f"Presently unsupported query format {suffix}")
            queries = Dataset.from_list([])
    return queries


def get_examples(examples_file_path: str) -> list[tuple[str, str]]:
    suffix = pathlib.Path(examples_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(examples_file_path, sep="\t")
            raw_queries = cast(
                Iterable[str],
                (
                    full_dataframe["query"]
                    if "query" in full_dataframe.columns
                    else full_dataframe["sentence"]
                ),
            )
            queries = (deserialize_whitespace(query) for query in raw_queries)
            responses = cast(Iterable[str], full_dataframe["response"])
            examples = list(zip(queries, responses))
        case ".txt" | "":
            examples = parse_input_output(examples_file_path)
        case _:
            ValueError(f"Presently unsupported examples file format {suffix}")
            examples = []
    return examples


def parse_input_output(examples_file_path: str) -> list[tuple[str, str]]:
    def parse_example(raw_example: str) -> tuple[str, str]:
        result = tuple(
            elem.strip()
            for elem in re.split("input:|output:", raw_example)
            if len(elem.strip()) > 0
        )
        assert len(result) == 2
        return result

    with open(examples_file_path, mode="rt", encoding="utf-8") as ef:
        no_comments_str = "".join(
            line for line in ef.readlines() if not line.strip().startswith("#")
        )
    return [
        parse_example(example.strip())
        # for example in no_comments_str.split("\n\n")
        for example in re.split("\n{2,}", no_comments_str)
        if len(example.split()) > 0
    ]


def get_document_level_example(
    sample_document_path: str, sample_answer_path: str
) -> tuple[str, str]:
    with open(sample_document_path, mode="rt", encoding="utf-8") as sample_document:
        # not normalizing newlines since those might be useful
        query = sample_document.read()
    sample_answer_dataframe = pd.read_csv(sample_answer_path, sep="\t")
    # specific to earlier use-case etc but for now
    answer = "\n".join(cast(Iterable[str], sample_answer_dataframe["query"]))
    return (query, answer)


def zero_shot_prompt(system_prompt: str, query: str) -> list[Message]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return messages


def few_shot_prompt(
    system_prompt: str, query: str, examples: Iterable[tuple[str, str]]
) -> list[Message]:
    def message_pair(ex_query: str, ex_answer: str) -> tuple[Message, ...]:
        return {"role": "user", "content": ex_query}, {
            "role": "assistant",
            "content": ex_answer,
        }

    few_shot_examples = chain.from_iterable(
        message_pair(ex_query=ex_query, ex_answer=ex_answer)
        for ex_query, ex_answer in examples
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        {"role": "user", "content": query},
    ]
    return messages


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


def main() -> None:
    args = parser.parse_args()
    process(
        args.model_name,
        args.text_column,
        args.prompt_file,
        args.query_files,
        args.max_new_tokens,
        args.output_dir,
        args.model_path,
        args.examples_file,
        args.sample_document,
        args.sample_answer,
        args.query_dir,
        args.batch_size,
        args.keeper_columns,
    )
if __name__ == "__main__":
    main()
