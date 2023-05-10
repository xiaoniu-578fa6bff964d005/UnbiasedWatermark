import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
# rouge = json.load(open('data/text_summarization_rouge.json'))
rouge=json.load(open('data/machine_translation_bleu.json'))

def map_wp_str(wp_str):
    if "Delta" in wp_str:
        return "$\delta$"
    elif "Gamma" in wp_str:
        return "$\gamma$"
    else:
        return wp_str
import pdb; pdb.set_trace()       
df = pd.DataFrame.from_dict(
    [{"wp": map_wp_str(wp_str), "score": score_str, "value": score} 
     for wp_str in rouge for score_str in rouge[wp_str] for score in ['precision', 'f1', 'recall']],
                            )


# import pdb; pdb.set_trace()                       
df.head()

Ns = []
gamma = []
delta = []

import seaborn as sns
sns.violinplot(data=df[df['score']=='rouge1'], x="wp", y="value",
order=["None", "$\gamma$", "$\delta$"])