import numpy as np
from sklearn.metrics import (precision_recall_curve,precision_score,recall_score,f1_score,
    average_precision_score,roc_auc_score,confusion_matrix,accuracy_score)

def threshold_from_validation(y,scores):
    y=np.asarray(y); scores=np.asarray(scores)
    if len(np.unique(y))!=2 or not np.isfinite(scores).all():
        raise ValueError('Validation must contain both classes and finite scores')
    p,r,t=precision_recall_curve(y,scores)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-12)
    best=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]
    return float(t[best])

def evaluate(y,scores,threshold):
    y=np.asarray(y); scores=np.asarray(scores); pred=scores>=threshold
    if len(np.unique(y))!=2 or not np.isfinite(scores).all():
        raise ValueError('Test must contain both classes and finite scores')
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'precision':float(precision_score(y,pred,zero_division=0)),
        'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),
        'average_precision':float(average_precision_score(y,scores)),
        'roc_auc':float(roc_auc_score(y,scores)),'accuracy':float(accuracy_score(y,pred)),
        'false_positive_rate':float(fp/(fp+tn)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
        'threshold':float(threshold),'threshold_rule':'maximum validation F1; highest threshold on ties',
        'pr_auc_convention':'average precision (AP), not trapezoidal area'}
