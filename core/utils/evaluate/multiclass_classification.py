import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix

from utils.io import write_json

def evaluate_multiclass_classification(dataset_dict,categories,csv_path,metric_path,plots_path):
    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(metric_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    # Generate the results dictionary with the postprocessed prediction
    results_dict = {'id': [],'gt': [], 'gt_index': [], 'pred': [], 'pred_index': []}

    for i, prediction in enumerate(dataset_dict['pred']):
        # Add gt and prediction to the results dictionary
        gt = dataset_dict['gt'][i]
        id=dataset_dict['input_path'][i]

        gt_index=np.argmax(gt)
        pred_index=np.argmax(prediction)

        results_dict['gt_index'].append(gt_index)
        results_dict['gt'].append(categories[gt_index])
        results_dict['pred_index'].append(pred_index)
        results_dict['pred'].append(categories[pred_index])
        results_dict['id'].append(id)
    
    # Calculate the accuracy and save it in a json
    accuracy = accuracy_score(results_dict['gt_index'], results_dict['pred_index'])
    recall = recall_score(results_dict['gt_index'], results_dict['pred_index'],average='weighted')
    precision=precision_score(results_dict['gt_index'], results_dict['pred_index'],average='weighted')
    f1= f1_score(results_dict['gt_index'], results_dict['pred_index'],average='weighted')
    metrics_dict = {'Accuracy': accuracy,
                    'Recall': recall,
                    'Precision': precision,
                    'F1_score': f1}

    write_json(os.path.join(metric_path,'metrics_.json' ), metrics_dict)

    # Create the csv with the results
    csv_file_path = os.path.join(csv_path, 'results.csv')
    df_complete = list()
    for i in range(len(results_dict['id'])):
        df_row = list()
        df_row.append(results_dict['id'][i])
        df_row.append(results_dict['gt'][i])
        df_row.append(results_dict['pred'][i])
        df_complete.append(df_row)
    test_results_df = pd.DataFrame(df_complete, columns=['image', 'GT', 'Prediction'])
    test_results_df.to_csv(csv_file_path,index=False)

    # CONFUSION MATRIX
    # Example data (replace with your actual data)
    gt_labels = results_dict['gt']
    pred_labels = results_dict['pred']

    # Create pandas Series from lists
    y_actu = pd.Series(gt_labels, name='GT')
    y_pred = pd.Series(pred_labels, name='Prediction')

    # Create the confusion matrix
    df_confusion = pd.crosstab(y_actu, y_pred)  

    # Plot the confusion matrix
    plt.figure(figsize=(4, 4))
    plt.imshow(df_confusion, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    # Display sample counts in each cell
    for i in range(len(df_confusion.index)):
        for j in range(len(df_confusion.columns)):
            plt.text(j, i, str(df_confusion.iloc[i, j]), ha='center', va='center', color='black')

    plt.xticks(range(len(df_confusion.columns)), df_confusion.columns, rotation=45)
    plt.yticks(range(len(df_confusion.index)), df_confusion.index)
    plt.xlabel('Prediction')
    plt.ylabel('GT')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path,'confusion_matrix.png'))