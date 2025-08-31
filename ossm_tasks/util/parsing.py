from typing import Any
from typing import Union

import xmlschema
import re
from ossm_tasks.config import SCHEMA_PATH


def _normalize_attribute_names(data: Union[dict, list, str]) -> Union[dict, list, str]:
    """  Recursively normalize attribute names by removing @ prefix. """

    def normalize(d: Union[dict, list, str]) -> Union:
        if isinstance(d, dict):
            return {k.lstrip('@'): normalize(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [normalize(item) for item in d]
        else:
            return d

    return normalize(data)

def parse_stef_xml(stef_file: str, schema_file: str = None) -> dict:
    """
    Validate and decode a STEF XML file via xmlschema.
    Strips namespace prefixes and always returns a clean dict under key 'task'.
    """
    schema = xmlschema.XMLSchema(schema_file or SCHEMA_PATH)
    schema.validate(stef_file)

    data = schema.to_dict(stef_file, validation='strict')
    parsed_data = {}

    top_level_elements = ['config', 'body', 'environment', 'objective', 'taskevent', 'action', 'termination']
    for element in top_level_elements:
        if element in data:
            parsed_data[element] = _normalize_attribute_names(data[element])

    return parsed_data