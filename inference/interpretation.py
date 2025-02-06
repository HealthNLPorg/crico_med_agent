import datetime
import os
from typing import Iterable

import pandas as pd
import pytz


def timestamp() -> str:
    current_time = datetime.datetime.now(pytz.timezone("America/New_York"))
    return current_time.strftime("%H:%M:%S  %m-%d-%Y")


class AnaforaEntity:
    def __init__(self, span: tuple[int, int], filename: str) -> None:
        self.anafora_id = -1
        self.span = span
        self.filename = filename

    def get_id_str(self) -> str:
        return f"{self.anafora_id}@e@{self.filename}@llama"

    def set_id(self, anafora_id) -> None:
        self.anafora_id = anafora_id


class Instruction(AnaforaEntity):
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


class Medication(AnaforaEntity):
    def __init__(self, span: tuple[int, int], filename: str) -> None:
        super().__init__(span, filename)
        self.instructions: list[Instruction] = []
        self.cui_str: str = ""
        self.tui_str: str = ""

    def __str__(self) -> str:
        # to return the XML entry
        instructions_str = "".join(
            f"<instruction_condition>{instruction.get_id_str()}</instruction_condition>"
            for instruction in self.instructions
        )

        properties_str = (
            "<negation_indicator/>\n"
            f"<associatedCode>{self.cui_str}</associatedCode>\n"
            f"<associatedTuiCodes>{self.tui_str}</associatedCode>\n"
            "<conditional/>\n"
            "<generic/>\n"
            "<subject/>\n"
            "<uncertainty_indicator/>\n"
            "<DocTimeRel/>\n"
            "<historyOf/>\n"
            "<allergy_indicator/>\n"
            "<change_status_model/>\n"
            "<dosage_model/>\n"
            "<duration_model/>\n"
            "<end_date/>\n"
            "<form_model/>\n"
            "<frequency_model/>\n"
            "<route_model/>\n"
        )
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0],self.span[1]}</span>\n"
            "<type>Medications/Drugs</type>\n"
            "<parentsType>UMLSEntities</parentsType>\n"
            "<properties>\n"
            "</properties>"
            "</entity>\n"
        )
        return ""

    def set_instructions(self, instructions: Iterable[Instruction]) -> None:
        self.instructions = list(instructions)

    def set_cui_str(self, cui_str: str) -> None:
        self.cui_str = cui_str

    def set_tui_str(self, tui_str: str) -> None:
        self.tui_str = tui_str


class AnaforaAnnotation:
    def __init__(self, entities: Iterable[AnaforaEntity]) -> None:
        self.entities = AnaforaAnnotation.order_entities(entities)

    def __str__(self) -> str:
        entities_str = "".join(str(entity) for entity in self.entities)
        return (
            "<data>\n"
            "<info>\n"
            f"<savetime>{timestamp()}</savetime>\n"
            "<progress>completed</progress>\n"
            '<schema path="./" protocol="file">temporal.schema.xml</schema>'
            "<annotations>\n"
            f"{entities_str}"
            "</annotations>"
            "</info>"
            "</data>"
        )

    @staticmethod
    def order_entities(entities: Iterable[AnaforaEntity]) -> list[AnaforaEntity]:
        def span(anafora_entity: AnaforaEntity) -> tuple[int, int]:
            return anafora_entity.span

        ordered_entities = sorted(entities, key=span, reverse=True)
        # Anafora XML starts from 1 not 0
        for anafora_id, anafora_entity in enumerate(ordered_entities, start=1):
            anafora_entity.set_id(anafora_id)
        return ordered_entities


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
