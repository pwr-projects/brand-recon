import os
from typing import Mapping
from xml.etree import ElementTree as ET

from .misc import isxml
from ..config import *


def get_annotations() -> Mapping[str, str]:
    files = os.listdir(PATH_ANNOTATIONS)
    files = map(lambda path: pj(PATH_ANNOTATIONS, path), filter(isxml, files))
    file_logo = {}
    
    for path in files: # tqdm(files, 'Parsing annotations', dynamic_ncols=True):
        with open(path, 'r') as fhd:
            header = "<?xml version='1.0' encoding='cp1252'?>"
            lines = fhd.readlines()

            if not lines[0].startswith(header):
                lines.insert(0, '{}\n'.format(header))

        with open(path, 'w') as fhd:
            fhd.writelines(lines)

        parsed = ET.parse(path).getroot()
        logo_name = parsed.findtext('.//object/name')
        filename = parsed.findtext('.//filename')
        file_logo[filename] = logo_name

    return file_logo
