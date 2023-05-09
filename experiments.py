#!/usr/bin/env python
# -*- coding: utf-8 -*-

from experiments_piece.text_summarization import *
from experiments_piece.machine_translation import *

if __name__ == "__main__":
    text_summarization_map()
    text_summarization_evaluate()
    machine_translation_map()
    machine_translation_evaluate()
