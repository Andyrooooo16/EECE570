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

def evaluate_model(model, test_loader, device, threshold=0.5):
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
                auc_per_class.append(auc)
            except ValueError:
                auc_per_class.append(float('nan'))
        
        metrics['roc_auc_per_class'] = auc_per_class
        metrics['macro_roc_auc'] = np.nanmean(auc_per_class)
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
        metrics['accuracy_per_class'].append(accuracy_score(y_true[:, i], y_pred[:, i]))
        metrics['precision_per_class'].append(precision_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics['recall_per_class'].append(recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
        metrics['f1_per_class'].append(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
        
        # Calculate Matthews Correlation Coefficient
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i])
            metrics['mcc_per_class'].append(mcc)
        except:
            metrics['mcc_per_class'].append(0.0)
        
        # Calculate TPR (Sensitivity) and TNR (Specificity)
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        metrics['tpr_per_class'].append(tpr)
        metrics['tnr_per_class'].append(tnr)
    
    # Calculate macro averages
    metrics['macro_accuracy'] = np.mean(metrics['accuracy_per_class'])
    metrics['macro_precision'] = np.mean(metrics['precision_per_class'])
    metrics['macro_recall'] = np.mean(metrics['recall_per_class'])
    metrics['macro_f1'] = np.mean(metrics['f1_per_class'])
    metrics['macro_mcc'] = np.mean(metrics['mcc_per_class'])
    metrics['macro_tpr'] = np.mean(metrics['tpr_per_class'])
    metrics['macro_tnr'] = np.mean(metrics['tnr_per_class'])
    
    return metrics

def plot_metrics(metrics, disease_cols, save_path="../outputs/Test/"):
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
    plt.axhline(y=metrics['macro_roc_auc'], color='r', linestyle='--', label=f'Macro ROC AUC: {metrics["macro_roc_auc"]:.3f}')
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
    plt.axhline(y=metrics['macro_f1'], color='r', linestyle='--', label=f'Macro F1: {metrics["macro_f1"]:.3f}')
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
    plt.axhline(y=metrics['macro_mcc'], color='r', linestyle='--', label=f'Macro MCC: {metrics["macro_mcc"]:.3f}')
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

def confusion_matrix_per_class(y_true, y_pred, disease_cols, save_path="../outputs/Test/"):
    """
    Calculate and save confusion matrix for each disease class
    """
    print("📊 Calculating confusion matrices...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create a CSV file for confusion matrices
    confusion_data = []
    
    for i, disease in enumerate(disease_cols):
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

def plot_top_performing_diseases(metrics, disease_cols, y_true, y_probas, top_n=7, save_path="../outputs/Test/"):
    """
    Plot ROC Curves and create F1 Score table for top N performing disease categories
    """
    print(f"📊 Analyzing top {top_n} performing disease categories...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get indices of top N diseases by F1 score
    top_indices = np.argsort(metrics['f1_per_class'])[-top_n:][::-1]
    top_diseases = [disease_cols[i] for i in top_indices]
    
    # Plot ROC curves for top diseases
    plt.figure(figsize=(10, 8))
    
    for idx in top_indices:
        disease = disease_cols[idx]
        fpr, tpr, _ = roc_curve(y_true[:, idx], y_probas[:, idx])
        auc = metrics['roc_auc_per_class'][idx]
        plt.plot(fpr, tpr, lw=2, label=f'{disease} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Top {top_n} Performing Diseases')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'top_{top_n}_roc_curves.png'))
    
    # Create F1 Score table for top diseases
    top_metrics = {
        'Disease': top_diseases,
        'F1_Score': [metrics['f1_per_class'][i] for i in top_indices],
        'Precision': [metrics['precision_per_class'][i] for i in top_indices],
        'Recall': [metrics['recall_per_class'][i] for i in top_indices],
        'ROC_AUC': [metrics['roc_auc_per_class'][i] for i in top_indices],
        'MCC': [metrics['mcc_per_class'][i] for i in top_indices],
        'TPR': [metrics['tpr_per_class'][i] for i in top_indices],
        'TNR': [metrics['tnr_per_class'][i] for i in top_indices]
    }
    
    # Save to CSV
    top_df = pd.DataFrame(top_metrics)
    top_df.to_csv(os.path.join(save_path, f'top_{top_n}_diseases.csv'), index=False)
    
    # Create a visual table
    plt.figure(figsize=(12, 6))
    plt.axis('tight')
    plt.axis('off')
    table = plt.table(cellText=top_df.values,
                      colLabels=top_df.columns,
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title(f'Top {top_n} Performing Disease Categories')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'top_{top_n}_table.png'))
    
    return top_diseases, top_indices

def plot_tsne_visualization(features, y_true, top_indices, disease_cols, save_path="../outputs/Test/"):
    """
    Generate t-SNE visualization for the top performing disease classes
    """
    print("🔄 Generating t-SNE visualization for top performing diseases...")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Get the features and labels for top performing diseases
    relevant_indices = []
    disease_labels = []
    
    # For each top disease, get samples that are positive for that disease
    for class_idx in top_indices:
        # Find samples positive for this disease
        positive_samples = np.where(y_true[:, class_idx] == 1)[0]
        relevant_indices.extend(positive_samples)
        disease_labels.extend([disease_cols[class_idx]] * len(positive_samples))
    
    # Make sure we have unique indices (a sample can have multiple diseases)
    unique_indices = list(set(relevant_indices))
    
    # Prepare the data for t-SNE
    selected_features = features[unique_indices]
    
    # Create new labels array for unique samples (each sample may have multiple diseases)
    sample_labels = []
    for idx in unique_indices:
        # Find which top diseases this sample has
        sample_diseases = [disease_cols[i] for i in top_indices if y_true[idx, i] == 1]
        if sample_diseases:
            # If sample has multiple diseases, join them with '+'
            sample_labels.append('+'.join(sample_diseases))
        else:
            # If no top diseases, mark as 'Other'
            sample_labels.append('Other')
    
    # Apply t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(selected_features)-1))
    tsne_features = tsne.fit_transform(selected_features)
    
    # Create a colormap for the top diseases
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
    
    plt.title('t-SNE Visualization of Top Performing Disease Classes')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'tsne_top_diseases.png'))
    
    print(f"✅ t-SNE visualization saved to {save_path}")

def main():
    # ----- Parameters -----
    test_image_folder = "../data/Test_Set/test_cropped"  
    test_csv_path = "../data/Test_Set/RFMiD_Testing_Labels.csv"  
    model_path = "fine_tuned_model.pth"  
    threshold = 0.5  # Classification threshold
    top_n = 7  # Number of top performing diseases to analyze
    
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
    confusion_matrix_per_class(y_true, y_pred, disease_cols)
    
    # ----- Analyze Top Performing Diseases -----
    top_diseases, top_indices = plot_top_performing_diseases(metrics, disease_cols, y_true, y_probas, top_n=top_n)
    
    # ----- Generate t-SNE Visualization -----
    plot_tsne_visualization(features, y_true, top_indices, disease_cols)
    
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
    
    # ----- Top Performing Disease Classes -----
    print(f"\n🏆 Top {top_n} performing disease classes:")
    for i, disease in enumerate(top_diseases):
        idx = disease_cols.index(disease)
        print(f"  {i+1}. {disease}")
        print(f"     F1 Score: {metrics['f1_per_class'][idx]:.4f}")
        print(f"     ROC AUC: {metrics['roc_auc_per_class'][idx]:.4f}")
        print(f"     MCC: {metrics['mcc_per_class'][idx]:.4f}")
    
    # ----- Identify Best and Worst Performing Classes -----
    best_class_idx = np.argmax(metrics['f1_per_class'])
    worst_class_idx = np.argmin(metrics['f1_per_class'])
    
    print(f"\n🏆 Best performing class: {disease_cols[best_class_idx]}")
    print(f"   F1 Score: {metrics['f1_per_class'][best_class_idx]:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc_per_class'][best_class_idx]:.4f}")
    print(f"   MCC: {metrics['mcc_per_class'][best_class_idx]:.4f}")
    print(f"   TPR: {metrics['tpr_per_class'][best_class_idx]:.4f}")
    print(f"   TNR: {metrics['tnr_per_class'][best_class_idx]:.4f}")
    
    print(f"\n⚠️ Worst performing class: {disease_cols[worst_class_idx]}")
    print(f"   F1 Score: {metrics['f1_per_class'][worst_class_idx]:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc_per_class'][worst_class_idx]:.4f}")
    print(f"   MCC: {metrics['mcc_per_class'][worst_class_idx]:.4f}")
    print(f"   TPR: {metrics['tpr_per_class'][worst_class_idx]:.4f}")
    print(f"   TNR: {metrics['tnr_per_class'][worst_class_idx]:.4f}")
    
    print("\n✅ Testing complete.")

if __name__ == "__main__":
    main()