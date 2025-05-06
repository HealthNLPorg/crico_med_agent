import datetime
import os
from enum import Enum
from typing import Iterable

import pytz
from lxml import etree
from lxml.etree import _Element  # for mypy


class Progress(Enum):
    completed = 0
    inprogress = 1

    def file_ext(self) -> str:
        match self:
            case Progress.completed:
                return "completed"
            case Progress.inprogress:
                return "inprogress"

    def xml_tag(self) -> str:
        match self:
            case Progress.completed:
                return "completed"
            case Progress.inprogress:
                return "in-progress"


def timestamp() -> str:
    current_time = datetime.datetime.now(pytz.timezone("America/New_York"))
    return current_time.strftime("%H:%M:%S %m-%d-%Y")


class AnaforaEntity:
    def __init__(
        self, span: tuple[int, int], filename: str, annotator: str = "llama"
    ) -> None:
        self.anafora_id = -1
        self.span = span
        self.filename = filename
        self.annotator = annotator

    def get_id_str(self) -> str:
        return (
            f"{self.anafora_id}@e@{self.filename}@{self.annotator}"
            if self.anafora_id > 0
            else "__NO_ID_ASSIGNED__"
        )

    def set_id(self, anafora_id) -> None:
        self.anafora_id = anafora_id


# In the annotations the properties
# field is of the form:
# <properties>
# 	<frequency_number></frequency_number>
# 	<frequency_unit>$(frequency unit anafora ID)</frequency_unit>
# </properties>
# but will leave empty
# for now since the model doesn't parse for frequency unit
class Frequency(AnaforaEntity):
    def __str__(self) -> str:
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0]},{self.span[1]}</span>\n"
            "<type>Frequency</type>\n"
            "<parentsType>Attributes_medication</parentsType>\n"
            "<properties/>\n"
            "</entity>\n"
        )


# Similar to frequency, in the annotations the properties
# field is of the form:
# <properties>
# 	<dosage_values>$(dosage value text)</dosage_values>
# </properties>
# but will leave empty
# for now since the model doesn't parse for that subannotation
class Dosage(AnaforaEntity):
    def __str__(self) -> str:
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0]},{self.span[1]}</span>\n"
            "<type>Dosage</type>\n"
            "<parentsType>Attributes_medication</parentsType>\n"
            "<properties/>\n"
            "</entity>\n"
        )


class Instruction(AnaforaEntity):
    def __str__(self) -> str:
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0]},{self.span[1]}</span>\n"
            "<type>Instruction</type>\n"
            "<parentsType>Attributes_medication</parentsType>\n"
            "<properties/>\n"
            "</entity>\n"
        )


class InstructionCondition(AnaforaEntity):
    def __str__(self) -> str:
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0]},{self.span[1]}</span>\n"
            "<type>InstructionCondition</type>\n"
            "<parentsType>Attributes_medication</parentsType>\n"
            "<properties/>\n"
            "</entity>\n"
        )


MedicationAttribute = Dosage | Frequency | Instruction | InstructionCondition


class Medication(AnaforaEntity):
    def __init__(self, span: tuple[int, int], filename: str) -> None:
        super().__init__(span, filename)
        self.instruction_conditions: list[InstructionCondition] = []
        self.instructions: list[Instruction] = []
        self.frequencies: list[Frequency] = []
        self.dosages: list[Dosage] = []
        self.cui_str: str = ""
        self.tui_str: str = ""

    def build_raw_string(self) -> str:
        def get_span(annotation: AnaforaEntity) -> tuple[int, int]:
            return annotation.span

        instruction_condition_str = (
            "".join(
                f"<instruction_condition>{instruction_condition.get_id_str()}</instruction_condition>\n"
                for instruction_condition in sorted(
                    self.instruction_conditions, key=get_span
                )
            )
            if len(self.instruction_conditions) > 0
            else "<instruction_condition/>\n"
        )

        instruction_str = (
            "".join(
                f"<instruction_>{instruction.get_id_str()}</instruction_>\n"
                for instruction in sorted(self.instructions, key=get_span)
            )
            if len(self.instructions) > 0
            else "<instruction_/>\n"
        )
        frequency_str = (
            "".join(
                f"<frequency_>{frequency.get_id_str()}</frequency_>\n"
                for frequency in sorted(self.frequencies, key=get_span)
            )
            if len(self.frequencies) > 0
            else "<frequency_/>\n"
        )
        dosage_str = (
            "".join(
                f"<dosage_>{dosage.get_id_str()}</dosage_>\n"
                for dosage in sorted(self.dosages, key=get_span)
            )
            if len(self.dosages) > 0
            else "<dosage_/>\n"
        )
        properties_str = (
            "<negation_indicator/>\n"
            f"<associatedCode>{self.cui_str}</associatedCode>\n"
            f"<associatedTuiCodes>{self.tui_str}</associatedTuiCodes>\n"
            "<conditional/>\n"
            "<generic/>\n"
            "<subject/>\n"
            "<uncertainty_indicator/>\n"
            "<DocTimeRel/>\n"
            "<historyOf/>\n"
            "<allergy_indicator/>\n"
            "<change_status_model/>\n"
            f"{dosage_str}"
            # "<dosage_model/>\n"
            "<duration_model/>\n"
            "<end_date/>\n"
            "<form_model/>\n"
            "<frequency_model/>\n"
            "<route_model/>\n"
            "<start_date/>\n"
            "<strength_model/>\n"
            f"{frequency_str}"
            # "<frequency_model_2/>\n"
            "<strength_model_2/>\n"
            f"{instruction_condition_str}"
            f"{instruction_str}"
        )
        return (
            "<entity>\n"
            f"<id>{self.get_id_str()}</id>\n"
            f"<span>{self.span[0]},{self.span[1]}</span>\n"
            "<type>Medications/Drugs</type>\n"
            "<parentsType>UMLSEntities</parentsType>\n"
            "<properties>\n"
            f"{properties_str}"
            "</properties>\n"
            "</entity>\n"
        )

    def __str__(self) -> str:
        return self.build_raw_string()

    def set_instructions(self, instructions: Iterable[Instruction]) -> None:
        self.instructions = list(instructions)

    def set_frequencies(self, frequencies: Iterable[Frequency]) -> None:
        self.frequencies = list(frequencies)

    def set_dosages(self, dosages: Iterable[Dosage]) -> None:
        self.dosages = list(dosages)

    def set_instruction_conditions(
        self, instruction_conditions: Iterable[InstructionCondition]
    ) -> None:
        self.instruction_conditions = list(instruction_conditions)

    def set_attributes(self, attributes: list[MedicationAttribute]) -> None:
        self.set_instructions(
            attribute for attribute in attributes if isinstance(attribute, Instruction)
        )
        self.set_instruction_conditions(
            attribute
            for attribute in attributes
            if isinstance(attribute, InstructionCondition)
        )
        self.set_frequencies(
            attribute for attribute in attributes if isinstance(attribute, Frequency)
        )
        self.set_dosages(
            attribute for attribute in attributes if isinstance(attribute, Dosage)
        )

    def set_cui_str(self, cui_str: str) -> None:
        self.cui_str = cui_str

    def set_tui_str(self, tui_str: str) -> None:
        self.tui_str = tui_str


class AnaforaDocument:
    def __init__(
        self,
        filename: str,
        schema: str = "crico",
        annotator: str = "llama",
        progress: Progress = Progress.inprogress,
    ) -> None:
        self.filename = filename
        self.schema = schema
        self.annotator = annotator
        self.progress = progress
        self.entities: list[AnaforaEntity] = []

    def set_entities(self, entities: Iterable[AnaforaEntity]) -> None:
        self.entities = AnaforaDocument.order_entities(entities)

    def get_out_fn(self) -> str:
        return f"{self.filename}.{self.schema}.{self.annotator}.{self.progress.file_ext()}.xml"

    def build_raw_string(self) -> str:
        entities_str = "".join(str(entity) for entity in self.entities)
        return (
            # '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<data>\n"
            "<info>\n"
            f"<savetime>{timestamp()}</savetime>\n"
            f"<progress>{self.progress.xml_tag()}</progress>\n"
            "</info>\n"
            '<schema path="./" protocol="file">temporal.schema.xml</schema>\n'
            "<annotations>\n"
            f"{entities_str}"
            "</annotations>\n"
            "</data>"
        )

    @staticmethod
    def order_entities(entities: Iterable[AnaforaEntity]) -> list[AnaforaEntity]:
        def span(anafora_entity: AnaforaEntity) -> tuple[int, int]:
            return anafora_entity.span

        ordered_entities = sorted(entities, key=span)
        # Anafora XML starts from 1 not 0
        for anafora_id, anafora_entity in enumerate(ordered_entities, start=1):
            anafora_entity.set_id(anafora_id)
        return ordered_entities

    def get_etree(self) -> _Element:
        return etree.fromstring(self.build_raw_string())

    def write_to_dir(self, output_dir: str) -> None:
        with open(os.path.join(output_dir, self.get_out_fn()), mode="wb") as f:
            f.write(etree.tostring(self.get_etree(), pretty_print=True))
