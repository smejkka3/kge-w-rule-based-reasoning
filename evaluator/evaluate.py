import pandas as pd
import numpy as np
import torch
from kge.model import KgeModel
from kge.util.io import load_checkpoint
import glob, os
from loguru import logger


class Evaluate():
    def __init__(self, candidates):
        self.X_candidates = candidates
        logger.info(f"Evaluating generated candidates...")

    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))\

    def find_closest(self, arr, val):
        idx = np.abs(arr - val).argmin()
        return idx

    def evaluate_and_return_facts(self):
        logger.info(f"Evaluating..")
        train_df = pd.read_csv("/home/smejkal/github/kge-w-rule-based-reasoning/data/train_data/train.txt", sep='\t',header=None,
                         names=None,
                         dtype=str,
                         usecols=[0, 1, 2])
                         
        # download link for this checkpoint given under results above
        latest_model = max(glob.glob(os.path.join("/home/smejkal/github/kge/local/experiments/", '*/')), key=os.path.getmtime) + "checkpoint_best.pt"
        checkpoint = load_checkpoint(latest_model)
        model = KgeModel.create_from(checkpoint)
        triples = []

        for triplet in self.X_candidates:
            s_idx = int(triplet[0])
            p_idx = int(triplet[1])
            o_idx = int(triplet[2])
            s = torch.Tensor([s_idx]).long()             # subject indexes
            p = torch.Tensor([p_idx]).long()             # relation indexes
            scores = model.score_sp(s, p)            # scores of all objects for (s,p,?)
            
            o = torch.Tensor([o_idx]).long()             #scores for relation
            scores_spo = model.score_spo(s, p, o)

            spo = scores_spo.detach().numpy()[0]
            scores = scores.detach().numpy()[0]
                                    
            idx = self.find_closest(scores, spo)

            scores_norm = self.NormalizeData(scores)

            if scores_norm[idx] > 0.9:
                logger.info(f"Adding triple: {str(model.dataset.entity_strings(s))[2:-2], str(model.dataset.relation_strings(p))[2:-2], str(model.dataset.entity_strings(o))[2:-2]} to the training dataset.")
                triples.append((str(model.dataset.entity_strings(s))[2:-2], str(model.dataset.relation_strings(p))[2:-2], str(model.dataset.entity_strings(o))[2:-2], scores_norm[idx]))

        return pd.DataFrame(triples)