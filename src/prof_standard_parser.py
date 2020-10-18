import json
from logging import getLogger
from pathlib import Path

from lxml import etree
from tqdm import tqdm

logger = getLogger(__name__)

# TODO: log standard name in message


def extract_standard_name(element):
    try:
        (standard_name,) = element.xpath("./NameProfessionalStandart/text()")
    except ValueError:
        logger.error("Error while extracting standard name")
        raise
    return standard_name


def extract_registration_number(element):
    try:
        (registration_number,) = element.xpath("./RegistrationNumber/text()")
    except ValueError:
        logger.error("Error while extracting registration number. It will be set to None")
        registration_number = None
    return registration_number


def extract_prof_activity(element):
    try:
        (prof_activity,) = element.xpath("./FirstSection/KindProfessionalActivity/text()")
    except ValueError:
        logger.error("Error while extracting prof. activity. It will be set to None")
        prof_activity = None
    return prof_activity


def extract_prof_activity_purpose(element):
    try:
        (prof_activity_purpose,) = element.xpath(
            "./FirstSection/PurposeKindProfessionalActivity/text()"
        )
    except ValueError:
        logger.error(
            "Error while extracting prof. activity purpose. It will be set to None"
        )
        prof_activity_purpose = None
    return prof_activity_purpose


def extract_general_work_function_name(element):
    try:
        (general_work_function_name,) = element.xpath("./NameOTF/text()")
    except ValueError:
        logger.error(
            "Error while extracting general work function name. It will be set to None"
        )
        general_work_function_name = None
    return general_work_function_name


def extract_general_work_function_code(element):
    try:
        (general_work_function_code,) = element.xpath("./CodeOTF/text()")
    except ValueError:
        logger.error(
            "Error while extracting general work function code. It will be set to None"
        )
        general_work_function_code = None
    return general_work_function_code


def extract_general_work_function_qualification(element):
    try:
        (qualification,) = element.xpath("./LevelOfQualification/text()")
    except ValueError:
        logger.error(
            "Error while extracting general work function qualification. "
            "It will be set to None"
        )
        qualification = None
    return qualification


def extract_particular_work_function_name(element):
    try:
        (particular_work_function_name,) = element.xpath("./NameTF/text()")
    except ValueError:
        logger.error(
            "Error while extracting particular work function name. It will be set to None"
        )
        particular_work_function_name = None
    return particular_work_function_name


def extract_particular_work_function_code(element):
    try:
        (particular_work_function_code,) = element.xpath("./CodeTF/text()")
    except ValueError:
        logger.error(
            "Error while extracting particular work function code. It will be set to None"
        )
        particular_work_function_code = None
    return particular_work_function_code


def extract_particular_work_function_qualification(element):
    try:
        (qualification,) = element.xpath("./SubQualification/text()")
    except ValueError:
        logger.error(
            "Error while extracting particular work function qualification. "
            "It will be set to None"
        )
        qualification = None
    return qualification


def extract_particular_work_functions(element):
    # TODO: change inner representation from list to dict

    particular_work_functions = []
    for particular_work_function in element.xpath(".//ParticularWorkFunction"):
        parsed_work_function = {}

        parsed_work_function["work_function_name"] = extract_particular_work_function_name(
            particular_work_function
        )
        parsed_work_function["work_function_code"] = extract_particular_work_function_code(
            particular_work_function
        )
        parsed_work_function[
            "subqualification"
        ] = extract_particular_work_function_qualification(particular_work_function)

        parsed_work_function["labor_actions"] = particular_work_function.xpath(
            ".//LaborAction/text()"
        )
        parsed_work_function["required_skills"] = particular_work_function.xpath(
            ".//RequiredSkill/text()"
        )
        parsed_work_function["necessary_knowledges"] = particular_work_function.xpath(
            ".//NecessaryKnowledge/text()"
        )
        particular_work_functions.append(parsed_work_function)
    return particular_work_functions


def extract_general_work_functions(element):
    # TODO: change inner representation from list to dict
    general_work_functions = []
    for gen_work_function in element.xpath(".//GeneralizedWorkFunction"):
        parsed_gen_work_function = {}
        parsed_gen_work_function["gen_function_name"] = extract_general_work_function_name(
            gen_work_function
        )
        parsed_gen_work_function["gen_function_code"] = extract_general_work_function_code(
            gen_work_function
        )
        parsed_gen_work_function[
            "qualification"
        ] = extract_general_work_function_qualification(gen_work_function)

        parsed_gen_work_function[
            "particular_work_functions"
        ] = extract_particular_work_functions(gen_work_function)

        general_work_functions.append(parsed_gen_work_function)

    return general_work_functions


def parse_xml(file_path):
    parser = etree.XMLParser(
        remove_blank_text=True,
        resolve_entities=False,
        encoding="utf-8",
        ns_clean=True,
        recover=True,
    )
    with open(file_path, encoding="utf-8") as f:
        tree = etree.parse(f, parser=parser)
    data = []
    for prof_standard in tree.iter("ProfessionalStandart"):
        parsed_element = {}
        parsed_element["standard_name"] = extract_standard_name(prof_standard)
        parsed_element["registration_number"] = extract_registration_number(prof_standard)
        parsed_element["prof_activity"] = extract_prof_activity(prof_standard)
        parsed_element["prof_activity_purpose"] = extract_prof_activity_purpose(
            prof_standard
        )

        parsed_element["general_work_functions"] = extract_general_work_functions(
            prof_standard
        )
        data.append(parsed_element)
    return data


# TODO: separate script
def process_folder(input_path, output_path):
    # TODO: check input and output paths

    standards = []
    for file in tqdm(Path(input_path).glob("*xml")):
        standards.extend(parse_xml(str(file)))
    with open(output_path, "w") as f:
        json.dump(standards, f, indent=4, ensure_ascii=False)
