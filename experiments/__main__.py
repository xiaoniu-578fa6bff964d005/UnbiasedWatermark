from . import text_summarization as ts
from . import machine_translation as mt


def add_watermark_exp():
    ts.get_output.pipeline()
    ts.evaluate.pipeline()
    ts.evaluate_ppl.pipeline()

    mt.get_output.pipeline()
    mt.evaluate.pipeline()
    mt.evaluate_bleu.compute_bleu()
    mt.evaluate_ppl.pipeline()


if __name__ == "__main__":
    add_watermark_exp()
    ts.evaluate_score.pipeline()
