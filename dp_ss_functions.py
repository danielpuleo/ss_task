import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

def get_characters_until_first_number(s):
    match = re.match(r'^[^\d]*\d', s)
    return match.group(0) if match else s

def diff_calculation(a,b,niter=1000,sampleSize=1000):
    diffMeans = []
    for i in range(niter):
        
        # seasonal
        A = np.random.choice(a,size=sampleSize)
        AMean = np.mean(A)
        
        # non seasonal
        B = np.random.choice(b,size=sampleSize)
        BMean = np.mean(B)

        diff = AMean - BMean
        diffMeans.append(diff)
        i = i + 1
    return diffMeans

def hdi(arr, prob=0.95):
    """
    Compute the HDI for input array
    :param arr: the input array
    :param prob: probability mass (default to 0.9)
    :return: tuple lower_bound, upper_bound
    """
    arr.sort()
    ci_index_inc = int(prob * len(arr))
    n_cis = len(arr) - ci_index_inc
    ci_width = []
    for i in range(n_cis):
        ci_width.append(arr[i + ci_index_inc] - arr[i])
    ci_width = np.array(ci_width)
    hdi_min = arr[np.argmin(ci_width)]
    hdi_max = arr[np.argmin(ci_width) + ci_index_inc]
    return hdi_min, hdi_max

def hdi_plot(diff_of_means,xlabel='',title=''):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(diff_of_means, bins = 1000_000, density=True, histtype='step', cumulative=-1,
            label='Reversed emp.',color='blue')
    plt.axvline(x=0,color='black',linestyle='--')
    plt.ylabel('Prob.')
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xlim(min(diff_of_means))
    hdi_min, hdi_max = hdi(diff_of_means)
    hdi_plottable = list(frange(hdi_min,hdi_max,(hdi_max-hdi_min)/100))
    plt.plot((hdi_plottable), np.zeros(len(hdi_plottable)), linewidth=8, color = 'black')

def frange(start, stop, step=1.0):
    ''' "range()" like function which accept float type'''
    i = start
    while i < stop:
        yield i
        i += step

def one_hot_encode(df,dropFirst=False):
    # Select only categorical columns (object or category type)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Apply one-hot encoding using pandas.get_dummies
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=dropFirst)
    
    return df_encoded

def kfold_cross_validation(X, y, model, n_splits=5, threshold=0.5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=12)
    
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    log_losses = []
    auc_scores = []
    
    fold = 1
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train the model
        model.fit(X_train, y_train)
        
        # For models that support predict_proba
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            
            # Apply the custom threshold for binary classification
            if probabilities.shape[1] == 2:  # Binary classification
                predictions = (probabilities[:, 1] >= threshold).astype(int)
                # AUC for binary classification
                auc = roc_auc_score(y_test, probabilities[:, 1])
            else:
                # For multi-class, assign class if the max probability exceeds the threshold
                predictions = np.argmax(probabilities, axis=1)
                max_probabilities = np.max(probabilities, axis=1)
                predictions = np.where(max_probabilities >= threshold, predictions, -1)  # -1 means "no confident prediction"
                # AUC for multi-class classification
                auc = roc_auc_score(y_test, probabilities, multi_class="ovr")
        else:
            # For models that don't support probabilities, use predict
            predictions = model.predict(X_test)
            auc = None  # AUC requires probabilities, so it's not applicable here
        
        if auc is not None:
            auc_scores.append(auc)
        
        # Calculate accuracy for this fold
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)
        
        # Calculate other metrics
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted',zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted',zero_division=0)
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        if hasattr(model, "predict_proba"):
            log_loss_value = log_loss(y_test, probabilities)
            log_losses.append(log_loss_value)
        
        fold += 1
    
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)
    
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Mean Precision: {mean_precision}")
    print(f"Mean Recall: {mean_recall}")
    print(f"Mean F1 Score: {mean_f1}")
    
    if auc_scores:
        mean_auc = np.mean(auc_scores)
        print(f"Mean AUC Score: {mean_auc}")
    
    if log_losses:
        mean_log_loss = np.mean(log_losses)
        print(f"Mean Log Loss: {mean_log_loss}")
    
    return {
        "accuracy": mean_accuracy,
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "log_loss": np.mean(log_losses) if log_losses else None,
        "auc": np.mean(auc_scores) if auc_scores else None
    }
