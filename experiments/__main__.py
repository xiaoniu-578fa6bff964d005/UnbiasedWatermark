from . import text_summarization as ts
from . import machine_translation as mt


def add_watermark_exp():
    print("ts.get_output.pipeline()")
    ts.get_output.pipeline()
    print("ts.evaluate.pipeline()")
    ts.evaluate.pipeline()
    print("ts.evaluate_ppl.pipeline()")
    ts.evaluate_ppl.pipeline()

    print("mt.get_output.pipeline()")
    mt.get_output.pipeline()
    print("mt.evaluate.pipeline()")
    mt.evaluate.pipeline()
    print("mt.evaluate_bleu.compute_bleu()")
    mt.evaluate_bleu.compute_bleu()
    print("mt.evaluate_ppl.pipeline()")
    mt.evaluate_ppl.pipeline()


if __name__ == "__main__":
    print("Add watermark experiment")
    add_watermark_exp()

    print("Evaluate score experiment")
    print("ts.evaluate_score.pipeline()")
    ts.evaluate_score.pipeline()
    print("mt.evaluate_score.pipeline()")
    mt.evaluate_score.pipeline()
    print("ts.evaluate_score.pipeline2()")
    ts.evaluate_score.pipeline2()
    print("mt.evaluate_score.pipeline2()")
    mt.evaluate_score.pipeline2()
