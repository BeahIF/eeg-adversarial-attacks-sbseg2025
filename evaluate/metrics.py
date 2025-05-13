import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, dataloader_or_data, labels=None, adversarial=False):
    """
    Avalia o modelo em dados limpos ou adversariais.

    - Se `dataloader_or_data` for um DataLoader, ele avalia o conjunto de validação padrão.
    - Se for um ndarray (ex: adversarial), usa diretamente os dados.
    """

    model.eval()
    all_preds = []
    all_targets = []

    if isinstance(dataloader_or_data, torch.utils.data.DataLoader):
        with torch.no_grad():
            for X_batch, y_batch in dataloader_or_data:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
    else:
        # Adversarial (X_adv, y_true)
        X_tensor = torch.tensor(dataloader_or_data, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.long)

        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            all_preds = predicted.cpu().numpy()
            all_targets = y_tensor.cpu().numpy()

    acc = accuracy_score(all_targets, all_preds)
    print(f"\n{'🔒 Adversarial' if adversarial else '✅ Normal'} Accuracy: {acc*100:.2f}%")
    print("Classification Report:\n", classification_report(all_targets, all_preds))
    print("Confusion Matrix:\n", confusion_matrix(all_targets, all_preds))
