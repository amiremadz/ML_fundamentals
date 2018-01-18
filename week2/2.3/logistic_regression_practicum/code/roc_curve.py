import numpy as np


def roc_curve(probabilities, labels):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: list, list, list

    Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''
    # Generating random threshold value between .5 to 1
    threshold=np.linspace(.5, 1, 5)
    # List for storing False Positive Rate , True Positive Rate and threshold
    fpr=[]
    tpr=[]
    thresholds=[]
    length=len(threshold)
    
    # Calculating True Positive,True Negative,False Positive,False Negative Values
    
    for i in range(length):        
        
        tp=np.logical_and( probabilities > threshold[i], labels == 1 ).sum()
        tn = np.logical_and ( probabilities <= threshold[i], labels == 0 ).sum()
        fp = np.logical_and ( probabilities > threshold[i], labels == 0 ).sum()
        fn = np.logical_and ( probabilities <= threshold[i], labels == 1 ).sum()
        fpr.append(fp / float(fp + tn))
        tpr.append(tp / float(tp + fn))
        thresholds.append(threshold[i])
    return tpr, fpr, thresholds
