from . import text_summarization as ts
from . import machine_translation as mt

if __name__ == "__main__":
    ts.get_output.pipeline()
    ts.evaluate.pipeline()
    mt.get_output.pipeline()
    mt.evaluate.pipeline()
