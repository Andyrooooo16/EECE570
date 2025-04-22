import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt

from model import RetinalDiseaseClassifier
from train_model import RetinalFundusDataset

def evaluate_model(model, test_loader, device, threshold=0.125):
    """Evaluate the model on test data and extract features"""
    print("🔍 Starting model evaluation...")
    model.eval()
    
    all_labels, all_outputs, all_predictions, all_features = [], [], [], []
    features_hook = []
    
    def hook_fn(module, input, output):
        features_hook.append(output.detach().cpu())
    
    # Register hook on appropriate layer
    if hasattr(model, 'fc'):
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, torch.nn.Sequential):
            hook_handle = model.features.register_forward_hook(hook_fn)
        else:
            hook_handle = model.classifier[0].register_forward_hook(hook_fn)
    else:
        children = list(model.children())
        hook_handle = children[-2 if len(children) >= 2 else -1].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.cpu()
            outputs = model(images).cpu()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).float()
            
            if features_hook:
                batch_features = features_hook.pop()
                if len(batch_features.shape) > 2:
                    batch_features = torch.mean(batch_features, dim=[2, 3])
                all_features.append(batch_features)
            
            all_labels.append(labels)
            all_outputs.append(probabilities)
            all_predictions.append(predictions)
    
    hook_handle.remove()
    
    y_true = torch.cat(all_labels).numpy()
    y_probas = torch.cat(all_outputs).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    features = torch.cat(all_features).numpy() if all_features else y_probas
    
    return y_true, y_probas, y_pred, features

def find_optimal_threshold(y_true, y_probas, disease_cols, selected_diseases=None, 
                           metric_weights={'f1': 1.0, 'mcc': 1.0, 'recall': 1.0}, 
                           save_path="../outputs/Evaluation/"):
    """Find optimal classification thresholds for each disease"""
    print("🔍 Finding optimal thresholds...")
    os.makedirs(save_path, exist_ok=True)
    
    # Normalize weights
    total_weight = sum(metric_weights.values())
    weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    indices = [disease_cols.index(d) for d in selected_diseases if d in disease_cols] if selected_diseases else range(len(disease_cols))
    optimal_thresholds = {}
    best_metrics = {}
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for idx in indices:
        disease = disease_cols[idx]
        disease_metrics = {'threshold': [], 'f1': [], 'mcc': [], 'recall': [], 'combined': []}
        
        for threshold in thresholds:
            y_pred = (y_probas[:, idx] >= threshold).astype(int)
            
            f1 = f1_score(y_true[:, idx], y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true[:, idx], y_pred)
            recall = recall_score(y_true[:, idx], y_pred, zero_division=0)
            combined_score = weights['f1'] * f1 + weights['mcc'] * mcc + weights['recall'] * recall
            
            disease_metrics['threshold'].append(threshold)
            disease_metrics['f1'].append(f1)
            disease_metrics['mcc'].append(mcc or 0.0)
            disease_metrics['recall'].append(recall)
            disease_metrics['combined'].append(combined_score)
        
        best_idx = np.argmax(disease_metrics['combined'])
        optimal_thresholds[disease] = round(disease_metrics['threshold'][best_idx], 2)
        
        best_metrics[disease] = {
            'threshold': optimal_thresholds[disease],
            'f1': round(disease_metrics['f1'][best_idx], 4),
            'mcc': round(disease_metrics['mcc'][best_idx], 4),
            'recall': round(disease_metrics['recall'][best_idx], 4),
            'combined_score': round(disease_metrics['combined'][best_idx], 4)
        }
    
    # Calculate average best threshold
    valid_diseases = [d for d in selected_diseases if d in disease_cols] if selected_diseases else disease_cols
    avg_threshold = round(np.mean([optimal_thresholds[d] for d in valid_diseases]), 4)
    print(f"\n📊 Average Best Threshold: {avg_threshold}")
    
    # Save results
    threshold_data = [
        {**{'Disease': disease, 'Optimal_Threshold': threshold}, **{f'{k.title()}': v for k, v in best_metrics[disease].items() if k != 'threshold'}}
        for disease, threshold in optimal_thresholds.items()
    ]
    
    threshold_df = pd.DataFrame(threshold_data)
    threshold_csv_path = os.path.join(save_path, 'optimal_thresholds.csv')
    threshold_df.to_csv(threshold_csv_path, index=False)
    
    with open(threshold_csv_path, 'a') as f:
        f.write(f"Average ({['Selected', 'All'][selected_diseases is None]} Diseases),{avg_threshold},,,,\n")
    
    print(f"💾 Optimal thresholds saved to {threshold_csv_path}")
    return optimal_thresholds, best_metrics

def calculate_metrics(y_true, y_probas, y_pred, disease_cols):
    """Calculate comprehensive performance metrics"""
    print("📊 Calculating performance metrics...")
    metrics = {}
    
    # Calculate per-class and macro metrics
    metrics_list = {
        'roc_auc': [], 'accuracy': [], 'precision': [], 'recall': [], 
        'f1': [], 'mcc': [], 'tpr': [], 'tnr': []
    }
    
    for i, disease in enumerate(disease_cols):
        # ROC AUC
        try:
            auc = roc_auc_score(y_true[:, i], y_probas[:, i])
            metrics_list['roc_auc'].append(round(auc, 4))
        except ValueError:
            metrics_list['roc_auc'].append(float('nan'))
        
        # Other metrics
        metrics_list['accuracy'].append(round(accuracy_score(y_true[:, i], y_pred[:, i]), 4))
        metrics_list['precision'].append(round(precision_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        metrics_list['recall'].append(round(recall_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        metrics_list['f1'].append(round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        
        # MCC
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
            metrics_list['mcc'].append(round(mcc, 4))
        except:
            metrics_list['mcc'].append(0.0)
        
        # TPR and TNR
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        metrics_list['tpr'].append(round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4))  # Sensitivity
        metrics_list['tnr'].append(round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4))  # Specificity
    
    # Store per-class metrics
    for metric_name, values in metrics_list.items():
        metrics[f'{metric_name}_per_class'] = values
        metrics[f'macro_{metric_name}'] = round(np.nanmean(values), 4)
    
    return metrics

def plot_metrics(metrics, disease_cols, save_path="../outputs/Evaluation/", selected_diseases=None):
    """Plot and save evaluation metrics"""
    print("📈 Plotting evaluation results...")
    os.makedirs(save_path, exist_ok=True)
    
    # Filter for selected diseases
    if selected_diseases:
        selected_indices = [i for i, disease in enumerate(disease_cols) if disease in selected_diseases]
        filtered_diseases = [disease_cols[i] for i in selected_indices]
    else:
        selected_indices = range(len(disease_cols))
        filtered_diseases = disease_cols
    
    # Get filtered metrics
    metrics_to_plot = {
        'ROC AUC': [metrics['roc_auc_per_class'][i] for i in selected_indices],
        'F1 Score': [metrics['f1_per_class'][i] for i in selected_indices],
        'MCC': [metrics['mcc_per_class'][i] for i in selected_indices]
    }
    
    # Plot metrics and save results
    for metric_name, values in metrics_to_plot.items():
        avg_value = round(np.nanmean(values), 4)
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(filtered_diseases)), values)
        plt.axhline(y=avg_value, color='r', linestyle='--', label=f'Average {metric_name}: {avg_value:.4f}')
        plt.xticks(range(len(filtered_diseases)), filtered_diseases, rotation=90)
        plt.xlabel('Disease Classes')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} per Disease Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{metric_name.lower().replace(' ', '_')}_per_class.png"))
        plt.close()
    
    # Save detailed results to CSV
    results_df = pd.DataFrame({
        'Disease': filtered_diseases,
        **{k.replace('_per_class', ''): [v[i] for i in selected_indices] 
           for k, v in metrics.items() if k.endswith('_per_class')}
    })
    results_df.to_csv(os.path.join(save_path, 'disease_metrics.csv'), index=False)
    
    # Save macro metrics
    macro_metrics = {
        'Metric': [k.replace('macro_', '').upper() for k in metrics.keys() if k.startswith('macro_')],
        'All Classes': [v for k, v in metrics.items() if k.startswith('macro_')],
        'Selected Diseases': [round(np.nanmean([metrics[f"{k.replace('macro_', '')}_per_class"][i] 
                                              for i in selected_indices]), 4)
                            for k in metrics.keys() if k.startswith('macro_')]
    }
    pd.DataFrame(macro_metrics).to_csv(os.path.join(save_path, 'macro_metrics.csv'), index=False)
    
    print(f"✅ Evaluation results saved to {save_path}")

def confusion_matrix_per_class(y_true, y_pred, disease_cols, save_path="../outputs/Evaluation/", selected_diseases=None):
    """Calculate and visualize confusion matrices"""
    print("📊 Calculating confusion matrices...")
    os.makedirs(save_path, exist_ok=True)
    
    selected_indices = [i for i, disease in enumerate(disease_cols) 
                        if selected_diseases is None or disease in selected_diseases]
    
    confusion_data = []
    for i in selected_indices:
        disease = disease_cols[i]
        
        # Calculate values
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        confusion_data.append({
            'Disease': disease,
            'True_Positive': tp,
            'False_Positive': fp,
            'True_Negative': tn,
            'False_Negative': fn
        })
        
        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        cm = np.array([[tn, fp], [fn, tp]])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {disease}')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{disease}.png'))
        plt.close()
    
    pd.DataFrame(confusion_data).to_csv(os.path.join(save_path, 'confusion_matrices.csv'), index=False)

def plot_selected_diseases_metrics(metrics, disease_cols, y_true, y_probas, selected_diseases, save_path="../outputs/Evaluation/"):
    """Plot ROC curves and create detailed tables for selected diseases"""
    print(f"📊 Analyzing selected disease categories...")
    os.makedirs(save_path, exist_ok=True)
    
    selected_indices = [disease_cols.index(disease) for disease in selected_diseases if disease in disease_cols]
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for idx in selected_indices:
        disease = disease_cols[idx]
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_probas[:, idx])
        auc = metrics['roc_auc_per_class'][idx]
        plt.plot(fpr, tpr, lw=2, label=f'{disease} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Selected Diseases')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'selected_diseases_roc_curves.png'))
    
    # Create metrics table
    selected_metrics = {
        'Disease': [disease_cols[i] for i in selected_indices],
        **{k.replace('_per_class', ''): [metrics[k][i] for i in selected_indices] 
           for k in metrics.keys() if k.endswith('_per_class')}
    }
    
    selected_df = pd.DataFrame(selected_metrics)
    selected_df.to_csv(os.path.join(save_path, 'selected_diseases_metrics.csv'), index=False)
    
    # Create visual table
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=selected_df.values,
                      colLabels=selected_df.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2] + [0.1] * (len(selected_df.columns) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Selected Disease Categories Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'selected_diseases_table.png'))
    
    return selected_indices

def main():
    # Parameters
    test_image_folder = "../data/Evaluation_Set/evaluation_cropped"
    test_csv_path = "../data/Evaluation_Set/RFMiD_Validation_Labels.csv"
    model_path = "../outputs/Training/best_fundus_model.pth"
    threshold = 0.125
    selected_diseases = ["DR", "MH", "ODC", "TSLN", "DN", "MYA", "ARMD", "BRVO", "ODP", "ODE"]
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Data preparation
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = RetinalFundusDataset(
        img_folder=test_image_folder,
        csv_path=test_csv_path,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    disease_cols = test_dataset.disease_cols
    
    # Load model
    model = RetinalDiseaseClassifier(num_classes=len(disease_cols), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Evaluation
    y_true, y_probas, y_pred, features = evaluate_model(model, test_loader, device, threshold)
    metrics = calculate_metrics(y_true, y_probas, y_pred, disease_cols)
    
    # Analysis and visualization
    plot_metrics(metrics, disease_cols, selected_diseases=selected_diseases)
    confusion_matrix_per_class(y_true, y_pred, disease_cols, selected_diseases=selected_diseases)
    selected_indices = plot_selected_diseases_metrics(metrics, disease_cols, y_true, y_probas, selected_diseases)
    
    # Find optimal thresholds
    optimal_thresholds, best_metrics = find_optimal_threshold(
        y_true=y_true,
        y_probas=y_probas,
        disease_cols=disease_cols,
        selected_diseases=selected_diseases,
        metric_weights={'f1': 1.0, 'mcc': 1.0, 'recall': 1.0}
    )
    
    # Print summary
    print("\n📋 EVALUATION SUMMARY:")
    for metric_name, value in {k: v for k, v in metrics.items() if k.startswith('macro_')}.items():
        print(f"{metric_name.replace('macro_', 'Macro ').title()}: {value:.4f}")
    
    # Print selected diseases performance
    print(f"\n🏥 Selected disease classes performance:")
    for disease in selected_diseases:
        if disease in disease_cols:
            idx = disease_cols.index(disease)
            print(f"\n  {disease}:")
            for metric in ['f1', 'roc_auc', 'precision', 'recall', 'mcc', 'tpr', 'tnr']:
                print(f"    {metric.upper()}: {metrics[f'{metric}_per_class'][idx]:.4f}")
            
            tn, fp, fn, tp = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1]).ravel()
            print(f"    Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        else:
            print(f"\n  {disease}: Not found in disease columns")
    
    # Average performance of selected diseases
    valid_selected_indices = [disease_cols.index(d) for d in selected_diseases if d in disease_cols]
    if valid_selected_indices:
        avg_metrics = {
            metric.replace('_per_class', '').upper(): round(np.nanmean([metrics[metric][i] for i in valid_selected_indices]), 4)
            for metric in metrics.keys() if metric.endswith('_per_class')
        }
        
        print("\n📊 Average performance across selected diseases:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()