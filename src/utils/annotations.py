import os
from os.path import join as pj
from xml.etree import ElementTree as ET

from ..config import *
from .misc import isxml

__all__ = [
    'get_annotations'
]


def get_annotations():
    files = os.listdir(PATH_ANNOTATIONS)
    files = map(lambda path: pj(PATH_ANNOTATIONS, path), filter(isxml, files))
    file_logo = {}

    for path in tqdm(files, 'Parsing annotations'):
        with open(path, 'r') as fhd:
            header = '<?xml version="1.0"?>'
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
