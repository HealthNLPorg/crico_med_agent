import pandas as pd
import os


class AnaforaAnnotation:
    def __init__(self, entities: list[AnaforaEntity]) -> None:
        self.entities = entities

    def __str__(self) -> str:
        return "<data>\n" "</data>"


class AnaforaEntity:
    def __init__(self, span: str, filename: str) -> None:
        self.anafora_id = -1
        self.span = span
        self.filename = filename

    def set_id(self, anafora_id) -> None:
        self.anafora_id = anafora_id


class Medication(AnaforaEntity):
    def __init__(self, span: str, filename: str) -> None:
        super().__init__(span, filename)
        self.instructions: list[Instruction] = []
        self.cuis: list[str] = []
        self.tuis: list[str] = []

    def __str__(self) -> str:
        # to return the XML entry
        instructions_str = "".join(
            f"<instruction_condition>{instruction.get_id_str()}</instruction_condition>"
            for instruction in self.instructions
        )
        return ""

    def set_instructions(self, instructions: list[Instruction]) -> None:
        self.instructions = instructions

    def set_cuis(self, cui_str: str) -> None:
        self.cuis = cui_str.split(",")

    def set_tuis(self, tui_str: str) -> None:
        self.tuis = tui_str.split(",")


class Instruction(AnaforaEntity):
    def get_id_str(self) -> str:
        return f"{self.anafora_id}@e@{self.filename}@llama"

    def __str__(self) -> str:
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0],self.span[1]}</span>\n"
            "<type>InstructionCondition</type>\n"
            "<parentsType>Attributes_medication</parentsType>\n"
            "<properties/>\n"
            "</entity>\n"
        )


def to_anafora_files(corpus_frame: pd.DataFrame, output_dir: str) -> None:
    for fn, fn_frame in corpus_frame.groupby(["filename"]):
        (base_fn,) = fn
        xml_base_fn = f"{base_fn}.llama.completed.xml"
        xml_path = os.path.join(output_dir, xml_base_fn)
        xml_content = to_xml_content(fn_frame)
        with open(xml_path, mode="wt") as xml_f:
            xml_f.write(xml_content)


def to_xml_content(fn_frame: pd.DataFrame) -> str:
    return ""


def parse_instruction(model_output: str) -> str:
    return ""
