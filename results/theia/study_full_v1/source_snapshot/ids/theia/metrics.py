"""Exact node matching; unknown labels excluded, never silently converted to zero."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score
from ids.metrics import threshold_from_validation


def select_threshold(y, scores, false_positive_budget=.01):
    y = np.asarray(y); scores = np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite validation scores')
    known = y >= 0; classes = set(y[known].tolist())
    if classes == {0,1}:
        return threshold_from_validation(y[known],scores[known]), 'maximum validation F1; highest threshold on ties'
    if classes == {0}:
        # Strict nextafter makes ties conservative with >= decisions.
        return float(np.nextafter(np.quantile(scores[known],1-false_positive_budget,method='higher'),np.inf)), 'validation negative quantile; no validation positives'
    return None, 'unavailable: validation has no labelled negatives; supply curated negatives or explicit scoped_background convention'


def evaluate(y, scores, threshold):
    y = np.asarray(y); scores = np.asarray(scores)
    if not np.isfinite(scores).all(): raise ValueError('Nonfinite scores')
    known = y >= 0; y = y[known]; scores = scores[known]
    m = {'known_targets':len(y),'unknown_targets':int((~known).sum()),
         'precision':None,'recall':None,'f1':None,'average_precision':None,'roc_auc':None,
         'tn':None,'fp':None,'fn':None,'tp':None,'false_positive_rate':None}
    if set(y.tolist()) == {0,1}:
        m['average_precision'] = float(average_precision_score(y,scores)); m['roc_auc'] = float(roc_auc_score(y,scores))
    if threshold is not None and len(y):
        pred = scores >= threshold; tn,fp,fn,tp = map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
        m.update(tn=tn,fp=fp,fn=fn,tp=tp,
                 precision=float(precision_score(y,pred,zero_division=0)),
                 recall=float(recall_score(y,pred,zero_division=0)) if tp+fn else None,
                 f1=float(f1_score(y,pred,zero_division=0)) if tp+fn else None,
                 false_positive_rate=fp/(fp+tn) if fp+tn else None)
    m['binary_metrics_defined'] = set(y.tolist()) == {0,1} and threshold is not None
    return m
