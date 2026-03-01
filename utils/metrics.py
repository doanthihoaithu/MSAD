import pandas as pd
from sklearn import metrics
import numpy as np
import math
import itertools, operator
import time

from sklearn.metrics import roc_curve, precision_recall_curve


# import matplotlib.pyplot as plt


class metricor:
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias 
    
    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score 
        else:
            scoreX = score
            
        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L

        
    def labels_conv(self, preds):
        '''return indices of predicted anomaly
        '''

        # p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        return index[0]
    
    def labels_conv_binary(self, preds):
        '''return predicted label
        '''
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p 

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue / MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0 
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1 
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1
        if score == 0:
            return 0
        else:
            return 1/score
        
    def b(self, i, length):
        bias = self.bias 
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1


    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)
    
    
    def metric_new(self, label, score, plot_ROC=False, alpha=0.2,coeff=3):
        '''input:
               Real labels and anomaly score in prediction
            
           output:
               AUC, 
               Precision, 
               Recall, 
               F-score, 
               Range-precision, 
               Range-recall, 
               Range-Fscore, 
               Precison@k, 
             
            k is chosen to be # of outliers in real labels
        '''
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None
        
        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None
        
        #area under curve
        auc = metrics.roc_auc_score(label, score)
        # plor ROC curve
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            # display.plot()            
            
        #precision, recall, F
        
        preds = score > (np.mean(score)+coeff*np.std(score))
        if np.sum(preds) == 0:
            preds = score > (np.mean(score)+2*np.std(score))
            if np.sum(preds) == 0:
                preds = score > (np.mean(score)+1*np.std(score))
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #range anomaly 
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]
        
        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        
        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))
        
        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k
        
        L = [auc, precision, recall, f, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k]
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_PR(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        # plt.figure()
        # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()
        AP = metrics.auc(recall, precision)
        #AP = metrics.average_precision_score(label, score)
        return precision, recall, AP
        
    def range_recall_new(self, labels, preds, alpha):   


        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)  
        range_label = self.range_convers_new(labels)
        
        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, p)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0

    def range_convers_new(self, label):
        '''
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        # return [(x[0][0], x[-1][0]) for x in [list(y) for (x,y) in itertools.groupby((enumerate(label)), operator.itemgetter(1)) if x == 1]]
        
        L = []
        i = 0
        j = 0 
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i+=1
                if i >= len(label):
                    break
            j = i+1
            # print('j'+str(j))
            if j >= len(label):
                if j==len(label):
                    L.append((i,j-1))
    
                break
            while label[j] != 0:
                j+=1
                if j >= len(label):
                    L.append((i,j-1))
                    break
            if j >= len(label):
                break
            L.append((i, j-1))
            i = j
        return L
        
        
    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair 
        preds predicted data
        '''

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score
    
    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count
    
    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            
            x1 = np.arange(e,min(e+window//2,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            l0 = int((e-s+1)*percentage)
            
            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
            
            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred
        
        TP = np.sum(product)
        
        # recall = min(TP/P,1)
        P_new = (P+np.sum(labels))/2      # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP/P_new,1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))
        
        
        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1]+1)]) > 0:
                existence += 1
                
        existence_ratio = existence/len(L)
        # print(existence_ratio)
        
        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall*existence_ratio
        
        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))
        
        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP/N_new
        
        Precision_RangeAUC = TP/np.sum(pred)
        
        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC
    
    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)
        
        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type=='window':
            labels = self.extend_postive_range(labels, window=window)
        else:   
            labels = self.extend_postive_range_individual(labels, percentage=percentage)
        
        TPR_list = [0]
        FPR_list = [0]
        Precision_list = [1]
        
        for i in np.linspace(0, len(score)-1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score>= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P, L)
            
            TPR_list.append(TPR)
            FPR_list.append(FPR)
            Precision_list.append(Precision)
            
        TPR_list.append(1)
        FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
        
        tpr = np.array(TPR_list)
        fpr = np.array(FPR_list)
        prec = np.array(Precision_list)
        
        width = fpr[1:] - fpr[:-1]
        height = (tpr[1:] + tpr[:-1])/2
        AUC_range = np.sum(width*height)
        
        width_PR = tpr[1:-1] - tpr[:-2]
        height_PR = (prec[1:] + prec[:-1])/2
        AP_range = np.sum(width_PR*height_PR)
        
        if plot_ROC:
            return AUC_range, AP_range, fpr, tpr, prec
        
        return AUC_range
        

    # TPR_FPR_window
    def RangeAUC_volume(self, labels_original, score, windowSize):
        score_sorted = -np.sort(-score)
        
        tpr_3d=[]
        fpr_3d=[]
        prec_3d=[]
        
        auc_3d=[]
        ap_3d=[]
        
        window_3d = np.arange(0, windowSize+1, 1)
        P = np.sum(labels_original)
       
        for window in window_3d:
            # print(window)
            # print(len(labels_original), sum(labels_original))
            labels = self.extend_postive_range(labels_original, window)
            # print(len(labels), sum(labels))
            # exit()
            
            L = self.range_convers_new(labels)
            
            
            TPR_list = [0]
            FPR_list = [0]
            Precision_list = [1]
            
            for i in np.linspace(0, len(score)-1, 250).astype(int):
                threshold = score_sorted[i]
                # print('thre='+str(threshold))
                pred = score>= threshold
                TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
                
                TPR_list.append(TPR)
                FPR_list.append(FPR)
                Precision_list.append(Precision)
                
            TPR_list.append(1)
            FPR_list.append(1)   # otherwise, range-AUC will stop earlier than (1,1)
            
            
            tpr = np.array(TPR_list)
            fpr = np.array(FPR_list)
            prec = np.array(Precision_list)
            
            tpr_3d.append(tpr)
            fpr_3d.append(fpr)
            prec_3d.append(prec)
            
            width = fpr[1:] - fpr[:-1]
            height = (tpr[1:] + tpr[:-1])/2
            AUC_range = np.sum(width*height)
            auc_3d.append(AUC_range)
            
            width_PR = tpr[1:-1] - tpr[:-2]
            height_PR = (prec[1:] + prec[:-1])/2
            AP_range = np.sum(width_PR*height_PR)
            ap_3d.append(AP_range)

        
        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d)/len(window_3d), sum(ap_3d)/len(window_3d)




def generate_curve(label,score,slidingWindow):
    """
    Computes the metrics 'VUS_ROC' and 'VUS_PR'

    :return avg_auc_3d: vus_roc
    :return avg_ap_3d: vus_pr
    """

    tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume(labels_original=label, score=score, windowSize=1*slidingWindow)

    return avg_auc_3d, avg_ap_3d

from abc import ABC
from typing import Optional, Tuple

import numpy as np
from scipy.special import kl_div


class InterpretabilityHitKScore:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self, top_k) -> None:
        self.top_k: Optional[int] = top_k

    def score(self, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> None:
        assert y_true_multivariate.ndim == 2
        assert y_score_per_var.ndim == 2
        # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
        # result = auc(fpr, tpr)

        y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)

        anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
        top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
        interpretability_list = []
        for labels, top_k_index in zip(y_true_multivariate, top_k_anomalous_dimension):
            if labels.sum() != 0.0:
                interpretability = labels[top_k_index].sum() / labels.sum()
                interpretability_list.append(interpretability)
            else:
                interpretability_list.append(np.nan)
        # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
        interpretability_scores = np.array(interpretability_list)

        return interpretability_scores[y_true == 1.0].mean()

    # def supports_continuous_scorings(self) -> bool:
    #     return True
    @property
    def name(self) -> str:
        return f'Interpretability_Hit_{self.top_k}_Score'.upper()

class InterpretabilityConditionalHitKScore:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self, top_k) -> None:
        self.top_k: Optional[int] = top_k

    def score_and_output_details(self, y_true_univariate: np.array, y_score_univariate: np.array, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> Tuple[float, float, float]:
        assert y_true_multivariate.ndim == 2
        assert y_score_per_var.ndim == 2
        assert y_true_univariate.ndim == 1
        assert y_score_univariate.ndim == 1
        # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
        # result = auc(fpr, tpr)

        y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
        assert (y_true == y_true_univariate).all()

        precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,2))
        f1_scores = 2 * (precision * recall) / (precision + recall)
        max_f1_score = np.max(f1_scores)
        optimal_threshold = thresholds[np.argmax(f1_scores)]

        if np.isnan(max_f1_score):
            max_f1_score = 0.0
            optimal_threshold = np.nan

        detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)

        anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
        top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
        interpretability_list = []
        for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension, detected_anomalies):
            if labels.sum() != 0.0 and detected_anomaly != 0.0:
                interpretability = labels[top_k_index].sum() / labels.sum()
                interpretability_list.append(interpretability)
            else:
                interpretability_list.append(0)
        # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
        interpretability_scores = np.array(interpretability_list)

        return interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold
        # all_results = []
        # for optimal_threshold in thresholds:
        #     detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
        #
        #     anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
        #     top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
        #     interpretability_list = []
        #     for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension, detected_anomalies):
        #         if labels.sum() != 0.0 and detected_anomaly != 0.0:
        #             interpretability = labels[top_k_index].sum() / labels.sum()
        #             interpretability_list.append(interpretability)
        #         else:
        #             interpretability_list.append(0)
        #     # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
        #     interpretability_scores = np.array(interpretability_list)
        #     all_results.append((interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold))
        #
        # all_results = np.array(all_results)
        # return np.trapz(all_results[:,0],all_results[:,2]), 0, 0

    # def supports_continuous_scorings(self) -> bool:
    #     return True
    @property
    def name(self) -> str:
        return f'Interpretability_Conditional_Hit_{self.top_k}_Score'.upper()

class InterpretabilityConditionalHitKScoreUpdate:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self, top_k) -> None:
        self.top_k: Optional[int] = top_k

    def score_and_output_details(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> Tuple[float, float, float]:
        assert y_true_multivariate.ndim == 2
        assert y_score_per_var.ndim == 2
        assert y_true_univariate.ndim == 1
        assert y_score_univariate.ndim == 1
        # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
        # result = auc(fpr, tpr)

        y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
        assert (y_true == y_true_univariate).all()

        precision, recall, thresholds = precision_recall_curve(y_true_univariate, np.round(y_score_univariate,2))
        # f1_scores = 2 * (precision * recall) / (precision + recall)
        # max_f1_score = np.max(f1_scores)
        # optimal_threshold = thresholds[np.argmax(f1_scores)]

        # if np.isnan(max_f1_score):
        #     max_f1_score = 0.0
        #     optimal_threshold = np.nan

        # detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
        #
        # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
        # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
        # interpretability_list = []
        # for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension, detected_anomalies):
        #     if labels.sum() != 0.0 and detected_anomaly != 0.0:
        #         interpretability = labels[top_k_index].sum() / labels.sum()
        #         interpretability_list.append(interpretability)
        #     else:
        #         interpretability_list.append(0)
        # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
        # interpretability_scores = np.array(interpretability_list)
        #
        # return interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold

        all_results = []
        for optimal_threshold in thresholds:
            y_pred = np.array(y_score_univariate >= optimal_threshold, dtype=float)
            result = self.calculate_interpretability_scores(y_pred, y_true_multivariate, y_score_per_var)
            all_results.append((result, 0, optimal_threshold))
            # detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
            #
            # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
            # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
            # interpretability_list = []
            # for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension,
            #                                                  detected_anomalies):
            #     if labels.sum() != 0.0 and detected_anomaly != 0.0:
            #         interpretability = labels[top_k_index].sum() / labels.sum()
            #         interpretability_list.append(interpretability)
            #     else:
            #         interpretability_list.append(0)
            # # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
            # interpretability_scores = np.array(interpretability_list)
            # all_results.append((interpretability_scores[y_true == 1.0].mean(), max_f1_score, optimal_threshold))

        all_results = np.array(all_results)
        return np.trapz(all_results[:, 0], all_results[:, 2]), 0, 0

    def calculate_interpretability_scores(self, y_pred, y_true_multivariate: np.ndarray, contribution_per_var: np.ndarray) -> np.ndarray:
        top_k_indices_matrix = contribution_per_var.argsort(axis=1)[:, -self.top_k:]
        interpretability_score_df = pd.DataFrame(columns=['score'])
        interpretability_score_df['score'] = np.zeros(y_pred.shape[0])
        interpretability_score_df['y_pred'] = y_pred
        interpretability_score_df['y_true'] = (y_true_multivariate.sum(axis=1) >= 1).astype(float)
        interpretability_score_df['detected_anomalous_dimension_count'] = np.take_along_axis(y_true_multivariate, top_k_indices_matrix, axis=1).sum(axis=1)
        interpretability_score_df['true_anomalous_dimension_count'] = y_true_multivariate.sum(axis=1)
        # interpretability_score_df.loc[interpretability_score_df['y_pred'] == 1.0, 'score'] = interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'detected_anomalous_dimension_count'] / interpretability_score_df.loc[interpretability_score_df['y_true'] == 1.0].loc[interpretability_score_df['y_pred'] == 1.0, 'true_anomalous_dimension_count']
        # interpretability_scores = np.zeros(y_pred.shape[0])
        detected_true_anomaly_index = interpretability_score_df[interpretability_score_df['y_pred'] == 1.0].loc[interpretability_score_df['y_true'] == 1.0].index
        interpretability_score_df.loc[detected_true_anomaly_index, 'score'] = interpretability_score_df.loc[detected_true_anomaly_index, 'detected_anomalous_dimension_count'].values / interpretability_score_df.loc[detected_true_anomaly_index,'true_anomalous_dimension_count'].values
        return interpretability_score_df[interpretability_score_df['y_true'] == 1.0]['score'].mean()

    # def supports_continuous_scorings(self) -> bool:
    #     return True
    @property
    def name(self) -> str:
        return f'Interpretability_Conditional_Hit_{self.top_k}_Score_Update'.upper()

# class InterpretabilityHitKScore:
#     """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.
#
#     Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
#     All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
#     The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
#     already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
#     classification results.
#     """
#
#     def __init__(self, top_k) -> None:
#         self.top_k: Optional[int] = top_k
#
#     def score(self, y_true_univariate, y_score_univariate, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> None:
#         assert y_true_multivariate.ndim == 2
#         assert y_score_per_var.ndim == 2
#         assert y_true_univariate.ndim == 1
#         assert y_score_univariate.ndim == 1
#         # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
#         # result = auc(fpr, tpr)
#
#         y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)
#         assert (y_true == y_true_univariate).all()
#
#         fpr, tpr, thresholds = roc_curve(y_true_univariate, y_score_univariate)
#         optimal_idx = np.argmax(tpr - fpr)
#         optimal_threshold = thresholds[optimal_idx]
#
#         detected_anomalies = np.array(y_score_univariate >= optimal_threshold, dtype=float)
#
#         anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
#         top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
#         interpretability_list = []
#         for labels, top_k_index, detected_anomaly in zip(y_true_multivariate, top_k_anomalous_dimension, detected_anomalies):
#             if labels.sum() != 0.0 and detected_anomaly != 0.0:
#                 interpretability = labels[top_k_index].sum() / labels.sum()
#                 interpretability_list.append(interpretability)
#             else:
#                 interpretability_list.append(0)
#         # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
#         interpretability_scores = np.array(interpretability_list)
#
#         return interpretability_scores[y_true == 1.0].mean()
#
#     # def supports_continuous_scorings(self) -> bool:
#     #     return True
#     @property
#     def name(self) -> str:
#         return f'Interpretability_Hit_{self.top_k}_Score'.upper()

def distribution_distance(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    assert ground_truth.ndim == 1
    assert prediction.ndim == 1

    return kl_div(ground_truth, prediction).sum()

    # distance = 0
    #
    # for distribution_true, distribution_predict in zip(ground_truth, prediction):
    #     distance += -(distribution_true*np.log(distribution_predict) + abs(distribution_true-distribution_predict)*np.log(abs(distribution_true-distribution_predict)))
    # return distance

class InterpretabilityLogScore:
    """Takes an anomaly scoring and ground truth labels to compute and apply a threshold to the scoring.

    Subclasses of this abstract base class define different strategies to put a threshold over the anomaly scorings.
    All strategies produce binary labels (0 or 1; 1 for anomalous) in the form of an integer NumPy array.
    The strategy :class:`~timeeval.metrics.thresholding.NoThresholding` is a special no-op strategy that checks for
    already existing binary labels and keeps them untouched. This allows applying the metrics on existing binary
    classification results.
    """

    def __init__(self, include_negative: bool) -> None:
        self.include_negative: Optional[int] = include_negative

    def score(self, y_true_multivariate: np.ndarray, y_score_per_var: np.ndarray) -> None:
        assert y_true_multivariate.ndim == 2
        assert y_score_per_var.ndim == 2
        # fpr, tpr, thresholds = roc_curve(y_true.reshape(-1), y_score.reshape(-1))
        # result = auc(fpr, tpr)

        y_true = (y_true_multivariate.sum(axis=1)>=1).astype(float)

        # anomaly_scores_per_var_ranking = np.argsort(y_score_per_var, axis=1)
        # top_k_anomalous_dimension = anomaly_scores_per_var_ranking[:, -self.top_k:]
        interpretability_list = []
        # smooth = 0.1
        y_score_per_var = np.where(y_score_per_var < 0, 0, y_score_per_var)
        y_score_per_var = y_score_per_var/y_score_per_var.sum(axis=1, keepdims=True)
        y_true_multivariate = y_true_multivariate + 0.1
        y_true_multivariate = y_true_multivariate/ y_true_multivariate.sum(axis=1, keepdims=True)
        for labels, anomaly_score_per_var, aggregated_label in zip(y_true_multivariate, y_score_per_var, y_true):
            # labels = softmax(labels)
            # anomaly_score_per_var = softmax(anomaly_score_per_var)
            interpretability = distribution_distance(labels, anomaly_score_per_var)
            if self.include_negative:
                interpretability_list.append(interpretability)
            else:
                if aggregated_label != 0.0:
                    interpretability_list.append(interpretability)
                else:
                    interpretability_list.append(np.nan)
        # interpretability_scores = np.sqrt(np.power(multivariate_labels-anomaly_scores_per_var,2).sum(axis=1))
        interpretability_scores = np.array(interpretability_list)

        return interpretability_scores[y_true == 1].mean()

    # def supports_continuous_scorings(self) -> bool:
    #     return True
    @property
    def name(self) -> str:
        return f'Interpretability_Log_Score'.upper()