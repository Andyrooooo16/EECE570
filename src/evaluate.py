import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, matthews_corrcoef, confusion_matrix, multilabel_confusion_matrix, precision_recall_curve
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model import RetinalDiseaseClassifier
from train_model import RetinalFundusDataset

def evaluate_model(model, test_loader, device, threshold=0.7):
    """
    Evaluate the model on test data
    """
    print("🔍 Starting model evaluation...")
    model.eval()
    
    all_labels = []
    all_outputs = []
    all_predictions = []
    all_features = []  # Store features for t-SNE plotting
    
    # Create a modified forward hook to capture features
    features_hook = []
    
    def hook_fn(module, input, output):
        features_hook.append(output.detach().cpu())
    
    # Register a forward hook on the layer before classification
    # You might need to adjust this depending on your model architecture
    if hasattr(model, 'fc'):
        # ResNet style: hook into the layer before the final fc
        hook_handle = model.avgpool.register_forward_hook(hook_fn)
    elif hasattr(model, 'classifier'):
        # VGG/DenseNet style
        if isinstance(model.classifier, torch.nn.Sequential):
            hook_handle = model.features.register_forward_hook(hook_fn)
        else:
            hook_handle = model.classifier[0].register_forward_hook(hook_fn)

    else:
        children = list(model.children())
        if len(children) >= 2:
            hook_handle = children[-2].register_forward_hook(hook_fn)
        elif len(children) > 0:
            hook_handle = children[-1].register_forward_hook(hook_fn)
            print("⚠️ Only one child module found, hooking into the last one.")
        else:
            raise RuntimeError("❌ Could not find a valid layer to register the hook.")

    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.cpu()
            outputs = model(images).cpu()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities >= threshold).float()
            
            # We'll get the features from the hook
            if features_hook:
                batch_features = features_hook.pop()
                # Flatten if needed
                if len(batch_features.shape) > 2:
                    batch_features = torch.mean(batch_features, dim=[2, 3])
                all_features.append(batch_features)
            
            all_labels.append(labels)
            all_outputs.append(probabilities)
            all_predictions.append(predictions)
    
    # Remove the hook when done
    hook_handle.remove()
    
    # Concatenate all batches
    y_true = torch.cat(all_labels).numpy()
    y_probas = torch.cat(all_outputs).numpy()
    y_pred = torch.cat(all_predictions).numpy()
    
    # Check if we have features before concatenating
    if all_features:
        features = torch.cat(all_features).numpy()
    else:
        # If feature extraction failed, create dummy features
        print("⚠️ Feature extraction was unsuccessful. Using output probabilities as fallback features.")
        features = y_probas
    
    return y_true, y_probas, y_pred, features

def find_optimal_threshold(y_true, y_probas, disease_cols, selected_diseases=None, metric_weights={'f1': 1.0, 'mcc': 1.0, 'recall': 1.0}):

    print("🔍 Finding optimal thresholds...")
    
    # Normalize weights to sum to 1
    total_weight = sum(metric_weights.values())
    weights = {k: v/total_weight for k, v in metric_weights.items()}
    
    if selected_diseases:
        indices = [disease_cols.index(d) for d in selected_diseases if d in disease_cols]
    else:
        indices = range(len(disease_cols))
    
    optimal_thresholds = {}
    best_metrics = {}
    
    # Test thresholds from 0.1 to 0.9 with step of 0.05
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for idx in indices:
        disease = disease_cols[idx]
        disease_metrics = {'threshold': [], 'f1': [], 'mcc': [], 'recall': [], 'combined': []}
        
        for threshold in thresholds:
            # Apply threshold to get binary predictions
            y_pred = (y_probas[:, idx] >= threshold).astype(int)
            
            # Calculate metrics
            f1 = f1_score(y_true[:, idx], y_pred, zero_division=0)
            try:
                mcc = matthews_corrcoef(y_true[:, idx], y_pred)
            except:
                mcc = 0.0
            recall = recall_score(y_true[:, idx], y_pred, zero_division=0)
            
            # Calculate weighted combination score
            combined_score = (weights['f1'] * f1 + 
                              weights['mcc'] * mcc + 
                              weights['recall'] * recall)
            
            # Store results
            disease_metrics['threshold'].append(threshold)
            disease_metrics['f1'].append(f1)
            disease_metrics['mcc'].append(mcc)
            disease_metrics['recall'].append(recall)
            disease_metrics['combined'].append(combined_score)
        
        # Find the threshold with maximum combined score
        best_idx = np.argmax(disease_metrics['combined'])
        optimal_thresholds[disease] = round(disease_metrics['threshold'][best_idx], 2)
        
        best_metrics[disease] = {
            'threshold': optimal_thresholds[disease],
            'f1': round(disease_metrics['f1'][best_idx], 4),
            'mcc': round(disease_metrics['mcc'][best_idx], 4),
            'recall': round(disease_metrics['recall'][best_idx], 4),
            'combined_score': round(disease_metrics['combined'][best_idx], 4)
        }
        
    return optimal_thresholds, best_metrics

def plot_threshold_metrics(y_true, y_probas, disease_cols, selected_diseases, save_path="../outputs/Evaluation/"):
    """
    Plot F1, MCC, and Recall metrics across different thresholds for selected diseases
    """
    print("📈 Plotting threshold metrics for selected diseases...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Use only a subset of thresholds for cleaner plots
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    for disease in selected_diseases:
        if disease in disease_cols:
            idx = disease_cols.index(disease)
            
            # Calculate metrics at each threshold
            f1_scores = []
            mcc_scores = []
            recall_scores = []
            
            for threshold in thresholds:
                y_pred = (y_probas[:, idx] >= threshold).astype(int)
                
                f1_scores.append(f1_score(y_true[:, idx], y_pred, zero_division=0))
                try:
                    mcc_scores.append(matthews_corrcoef(y_true[:, idx], y_pred))
                except:
                    mcc_scores.append(0.0)
                recall_scores.append(recall_score(y_true[:, idx], y_pred, zero_division=0))
            
            # Plot metrics vs thresholds
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, f1_scores, 'b-', label='F1 Score')
            plt.plot(thresholds, mcc_scores, 'g-', label='MCC')
            plt.plot(thresholds, recall_scores, 'r-', label='Recall')
            
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title(f'Performance Metrics vs Threshold for {disease}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path, f'threshold_metrics_{disease}.png'))
            plt.close()

def calculate_metrics(y_true, y_probas, y_pred, disease_cols):
    """
    Calculate performance metrics
    """
    print("📊 Calculating performance metrics...")
    metrics = {}
    
    # Calculate ROC AUC for each disease
    try:
        # Try to calculate ROC AUC per class
        auc_per_class = []
        for i, disease in enumerate(disease_cols):
            # Some classes might not have both positive and negative examples
            try:
                auc = roc_auc_score(y_true[:, i], y_probas[:, i])
                auc_per_class.append(round(auc, 4))
            except ValueError:
                auc_per_class.append(float('nan'))
        
        metrics['roc_auc_per_class'] = auc_per_class
        metrics['macro_roc_auc'] = round(np.nanmean(auc_per_class), 4)
    except Exception as e:
        print(f"⚠️ Error calculating ROC AUC: {e}")
        metrics['roc_auc_per_class'] = [float('nan')] * len(disease_cols)
        metrics['macro_roc_auc'] = float('nan')
    
    # Calculate other metrics
    metrics['accuracy_per_class'] = []
    metrics['precision_per_class'] = []
    metrics['recall_per_class'] = []
    metrics['f1_per_class'] = []
    metrics['mcc_per_class'] = []  # Matthews Correlation Coefficient
    metrics['tpr_per_class'] = []  # True Positive Rate (Sensitivity)
    metrics['tnr_per_class'] = []  # True Negative Rate (Specificity)
    
    for i, disease in enumerate(disease_cols):
        metrics['accuracy_per_class'].append(round(accuracy_score(y_true[:, i], y_pred[:, i]), 4))
        metrics['precision_per_class'].append(round(precision_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        metrics['recall_per_class'].append(round(recall_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        metrics['f1_per_class'].append(round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 4))
        
        # Calculate Matthews Correlation Coefficient
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
            metrics['mcc_per_class'].append(round(mcc, 4))
        except:
            metrics['mcc_per_class'].append(0.0000)
        
        # Calculate TPR (Sensitivity) and TNR (Specificity)
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        metrics['tpr_per_class'].append(round(tpr, 4))
        metrics['tnr_per_class'].append(round(tnr, 4))
    
    # Calculate macro averages
    metrics['macro_accuracy'] = round(np.mean(metrics['accuracy_per_class']), 4)
    metrics['macro_precision'] = round(np.mean(metrics['precision_per_class']), 4)
    metrics['macro_recall'] = round(np.mean(metrics['recall_per_class']), 4)
    metrics['macro_f1'] = round(np.mean(metrics['f1_per_class']), 4)
    metrics['macro_mcc'] = round(np.mean(metrics['mcc_per_class']), 4)
    metrics['macro_tpr'] = round(np.mean(metrics['tpr_per_class']), 4)
    metrics['macro_tnr'] = round(np.mean(metrics['tnr_per_class']), 4)
    
    return metrics

def plot_metrics(metrics, disease_cols, save_path="../outputs/Evaluation/"):
    """
    Plot and save evaluation metrics
    """
    print("📈 Plotting evaluation results...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Plot ROC AUC per class
    plt.figure(figsize=(12, 8))
    auc_values = metrics['roc_auc_per_class']
    plt.bar(range(len(disease_cols)), auc_values)
    plt.axhline(y=metrics['macro_roc_auc'], color='r', linestyle='--', label=f'Macro ROC AUC: {metrics["macro_roc_auc"]:.4f}')
    plt.xticks(range(len(disease_cols)), disease_cols, rotation=90)
    plt.xlabel('Disease Classes')
    plt.ylabel('ROC AUC Score')
    plt.title('ROC AUC Score per Disease Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'roc_auc_per_class.png'))
    
    # Plot F1 Score per class
    plt.figure(figsize=(12, 8))
    f1_values = metrics['f1_per_class']
    plt.bar(range(len(disease_cols)), f1_values)
    plt.axhline(y=metrics['macro_f1'], color='r', linestyle='--', label=f'Macro F1: {metrics["macro_f1"]:.4f}')
    plt.xticks(range(len(disease_cols)), disease_cols, rotation=90)
    plt.xlabel('Disease Classes')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Disease Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'f1_per_class.png'))
    
    # Plot MCC Score per class
    plt.figure(figsize=(12, 8))
    mcc_values = metrics['mcc_per_class']
    plt.bar(range(len(disease_cols)), mcc_values)
    plt.axhline(y=metrics['macro_mcc'], color='r', linestyle='--', label=f'Macro MCC: {metrics["macro_mcc"]:.4f}')
    plt.xticks(range(len(disease_cols)), disease_cols, rotation=90)
    plt.xlabel('Disease Classes')
    plt.ylabel('Matthews Correlation Coefficient')
    plt.title('MCC Score per Disease Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mcc_per_class.png'))
    
    # Create a summary table of macro metrics
    macro_metrics = {
        'Metric': ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'TPR (Sensitivity)', 'TNR (Specificity)'],
        'Value': [
            metrics['macro_roc_auc'],
            metrics['macro_accuracy'],
            metrics['macro_precision'],
            metrics['macro_recall'],
            metrics['macro_f1'],
            metrics['macro_mcc'],
            metrics['macro_tpr'],
            metrics['macro_tnr']
        ]
    }
    
    # Save results to CSV
    print("💾 Saving results to CSV...")
    results_df = pd.DataFrame({
        'Disease': disease_cols,
        'ROC_AUC': metrics['roc_auc_per_class'],
        'Accuracy': metrics['accuracy_per_class'],
        'Precision': metrics['precision_per_class'],
        'Recall': metrics['recall_per_class'],
        'F1_Score': metrics['f1_per_class'],
        'MCC': metrics['mcc_per_class'],
        'TPR': metrics['tpr_per_class'],
        'TNR': metrics['tnr_per_class']
    })
    results_df.to_csv(os.path.join(save_path, 'disease_metrics.csv'), index=False)
    
    # Save macro metrics
    pd.DataFrame(macro_metrics).to_csv(os.path.join(save_path, 'macro_metrics.csv'), index=False)
    
    print(f"✅ Evaluation results saved to {save_path}")

def confusion_matrix_per_class(y_true, y_pred, disease_cols, save_path="../outputs/Evaluation/", selected_diseases=None):
    """
    Calculate and save confusion matrix for selected disease classes
    """
    print("📊 Calculating confusion matrices for selected diseases...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create a CSV file for confusion matrices
    confusion_data = []
    
    # Filter for selected diseases if provided
    if selected_diseases:
        selected_indices = [i for i, disease in enumerate(disease_cols) if disease in selected_diseases]
    else:
        selected_indices = range(len(disease_cols))
    
    for i in selected_indices:
        disease = disease_cols[i]
        
        # Calculate confusion matrix values
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
        
        # Visualize confusion matrix for this disease
        plt.figure(figsize=(6, 5))
        cm = np.array([[tn, fp], [fn, tp]])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix: {disease}')
        plt.colorbar()
        plt.xticks([0, 1], ['Negative', 'Positive'])
        plt.yticks([0, 1], ['Negative', 'Positive'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add text annotations to the confusion matrix
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{disease}.png'))
        plt.close()
    
    # Save confusion matrices to CSV
    confusion_df = pd.DataFrame(confusion_data)
    confusion_df.to_csv(os.path.join(save_path, 'confusion_matrices.csv'), index=False)

def plot_selected_diseases_metrics(metrics, disease_cols, y_true, y_probas, selected_diseases, save_path="../outputs/Evaluation/"):
    """
    Plot ROC Curves and create detailed metrics table for selected disease categories
    """
    print(f"📊 Analyzing selected disease categories...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get indices of selected diseases
    selected_indices = [disease_cols.index(disease) for disease in selected_diseases if disease in disease_cols]
    
    # Plot ROC curves for selected diseases
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
    plt.title(f'ROC Curves for Selected Diseases')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'selected_diseases_roc_curves.png'))
    
    # Create metrics table for selected diseases
    selected_metrics = {
        'Disease': [disease_cols[i] for i in selected_indices],
        'F1_Score': [metrics['f1_per_class'][i] for i in selected_indices],
        'Precision': [metrics['precision_per_class'][i] for i in selected_indices],
        'Recall': [metrics['recall_per_class'][i] for i in selected_indices],
        'ROC_AUC': [metrics['roc_auc_per_class'][i] for i in selected_indices],
        'MCC': [metrics['mcc_per_class'][i] for i in selected_indices],
        'TPR': [metrics['tpr_per_class'][i] for i in selected_indices],
        'TNR': [metrics['tnr_per_class'][i] for i in selected_indices]
    }
    
    # Save to CSV
    selected_df = pd.DataFrame(selected_metrics)
    selected_df.to_csv(os.path.join(save_path, f'selected_diseases_metrics.csv'), index=False)
    
    # Create a visual table
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=selected_df.values,
                      colLabels=selected_df.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(f'Selected Disease Categories Performance Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'selected_diseases_table.png'))
    
    return selected_indices

def plot_tsne_visualization(features, y_true, selected_indices, disease_cols, save_path="../outputs/Evaluation/"):
    """
    Generate t-SNE visualization for the selected disease classes
    """
    print("🔄 Generating t-SNE visualization for selected diseases...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get the features and labels for selected diseases
    relevant_indices = []
    disease_labels = []
    
    # For each selected disease, get samples that are positive for that disease
    for class_idx in selected_indices:
        # Find samples positive for this disease
        positive_samples = np.where(y_true[:, class_idx] == 1)[0]
        relevant_indices.extend(positive_samples)
        disease_labels.extend([disease_cols[class_idx]] * len(positive_samples))
    
    # Make sure we have unique indices (a sample can have multiple diseases)
    unique_indices = list(set(relevant_indices))
    
    # Prepare the data for t-SNE
    if len(unique_indices) > 0:
        selected_features = features[unique_indices]
        
        # Create new labels array for unique samples (each sample may have multiple diseases)
        sample_labels = []
        for idx in unique_indices:
            # Find which selected diseases this sample has
            sample_diseases = [disease_cols[i] for i in selected_indices if y_true[idx, i] == 1]
            if sample_diseases:
                # If sample has multiple diseases, join them with '+'
                sample_labels.append('+'.join(sample_diseases))
            else:
                # If no selected diseases, mark as 'Other'
                sample_labels.append('Other')
        
        # Apply t-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(selected_features)-1))
        tsne_features = tsne.fit_transform(selected_features)
        
        # Create a colormap for the selected diseases
        unique_labels = list(set(sample_labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: color for label, color in zip(unique_labels, colors)}
        
        # Create t-SNE plot
        plt.figure(figsize=(12, 10))
        
        # Plot each unique label with its color
        for label in unique_labels:
            # Find indices for this label
            mask = np.array(sample_labels) == label
            plt.scatter(
                tsne_features[mask, 0], 
                tsne_features[mask, 1],
                c=[color_map[label]],
                label=label,
                alpha=0.7,
                s=50
            )
        
        plt.title('t-SNE Visualization of Selected Disease Classes')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'tsne_selected_diseases.png'))
        
        print(f"✅ t-SNE visualization saved to {save_path}")
    else:
        print("⚠️ No samples found for the selected diseases. Skipping t-SNE visualization.")

def main():
    # ----- Parameters -----
    test_image_folder = "../data/Evaluation_Set/evaluation_cropped"  # Update with your test data path
    test_csv_path = "../data/Evaluation_Set/RFMiD_Validation_Labels.csv"  # Update with your test labels path
    model_path = "../outputs/Training/best_fundus_model.pth"  # Path to your saved model
    threshold = 0.7  # Classification threshold
    
    # Define specific diseases for evaluation
    selected_diseases = ["DR", "MH", "ODC", "TSLN", "DN", "MYA", "ARMD", "BRVO", "ODP", "ODE"]
    
    # ----- Setup -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # ----- Test Transform -----
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # ----- Load Test Dataset -----
    print("🔄 Loading test dataset...")
    test_dataset = RetinalFundusDataset(
        img_folder=test_image_folder,
        csv_path=test_csv_path,
        transform=test_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    disease_cols = test_dataset.disease_cols
    
    # ----- Load Model -----
    print("🔄 Loading model...")
    model = RetinalDiseaseClassifier(num_classes=len(disease_cols), pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # ----- Evaluate Model -----
    y_true, y_probas, y_pred, features = evaluate_model(model, test_loader, device, threshold)
    
    # ----- Calculate Metrics -----
    metrics = calculate_metrics(y_true, y_probas, y_pred, disease_cols)
    
    # ----- Plot and Save Results -----
    plot_metrics(metrics, disease_cols)
    confusion_matrix_per_class(y_true, y_pred, disease_cols, selected_diseases=selected_diseases)
    
    # ----- Analyze Selected Diseases -----
    selected_indices = plot_selected_diseases_metrics(metrics, disease_cols, y_true, y_probas, selected_diseases)
    
    # ----- Generate t-SNE Visualization -----
    plot_tsne_visualization(features, y_true, selected_indices, disease_cols)
    
    # ----- Print Summary -----
    print("\n📋 EVALUATION SUMMARY:")
    print(f"Macro ROC AUC: {metrics['macro_roc_auc']:.4f}")
    print(f"Macro Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Macro MCC: {metrics['macro_mcc']:.4f}")
    print(f"Macro TPR (Sensitivity): {metrics['macro_tpr']:.4f}")
    print(f"Macro TNR (Specificity): {metrics['macro_tnr']:.4f}")
    
    # ----- Selected Disease Classes Performance -----
    print(f"\n🏥 Selected disease classes performance:")
    for disease in selected_diseases:
        if disease in disease_cols:
            idx = disease_cols.index(disease)
            print(f"\n  {disease}:")
            print(f"    F1 Score: {metrics['f1_per_class'][idx]:.4f}")
            print(f"    ROC AUC: {metrics['roc_auc_per_class'][idx]:.4f}")
            print(f"    Precision: {metrics['precision_per_class'][idx]:.4f}")
            print(f"    Recall: {metrics['recall_per_class'][idx]:.4f}")
            print(f"    MCC: {metrics['mcc_per_class'][idx]:.4f}")
            print(f"    TPR (Sensitivity): {metrics['tpr_per_class'][idx]:.4f}")
            print(f"    TNR (Specificity): {metrics['tnr_per_class'][idx]:.4f}")
            
            # Get confusion matrix values
            tn, fp, fn, tp = confusion_matrix(y_true[:, idx], y_pred[:, idx], labels=[0, 1]).ravel()
            print(f"    Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        else:
            print(f"\n  {disease}: Not found in disease columns")
    
    # ----- Average Performance of Selected Diseases -----
    valid_selected_indices = [disease_cols.index(disease) for disease in selected_diseases if disease in disease_cols]
    
    if valid_selected_indices:
        avg_metrics = {
            'ROC AUC': round(np.mean([metrics['roc_auc_per_class'][i] for i in valid_selected_indices]), 4),
            'Accuracy': round(np.mean([metrics['accuracy_per_class'][i] for i in valid_selected_indices]), 4),
            'Precision': round(np.mean([metrics['precision_per_class'][i] for i in valid_selected_indices]), 4),
            'Recall': round(np.mean([metrics['recall_per_class'][i] for i in valid_selected_indices]), 4),
            'F1 Score': round(np.mean([metrics['f1_per_class'][i] for i in valid_selected_indices]), 4),
            'MCC': round(np.mean([metrics['mcc_per_class'][i] for i in valid_selected_indices]), 4),
            'TPR': round(np.mean([metrics['tpr_per_class'][i] for i in valid_selected_indices]), 4),
            'TNR': round(np.mean([metrics['tnr_per_class'][i] for i in valid_selected_indices]), 4)
        }
        
        print("\n📊 Average performance across all selected diseases:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n✅ Evaluation complete.")

if __name__ == "__main__":
    main()