import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

def prepare_dataloaders(X, y, batch_size=32, train_split=0.8, shuffle=True):
    """
    Cria os DataLoaders para treino e validação.

    Args:
        X (np.ndarray): Dados de entrada normalizados.
        y (np.ndarray): Rótulos (int).
        batch_size (int): Tamanho do batch.
        train_split (float): Proporção de treino (0.8 = 80% treino, 20% val).
        shuffle (bool): Se deve embaralhar os dados no treino.

    Returns:
        Tuple[DataLoader, DataLoader]: DataLoader de treino e validação.
    """
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader, device='cpu', lr=0.001, epochs=60, patience=5, save_path='EEGNet_best_model.pth'):
    """
    Treina o modelo com early stopping.

    Args:
        model (nn.Module): Modelo EEGNet.
        train_loader (DataLoader): Dados de treino.
        val_loader (DataLoader): Dados de validação.
        device (str): 'cpu' ou 'cuda'.
        lr (float): Taxa de aprendizado.
        epochs (int): Número máximo de épocas.
        patience (int): Número de épocas para early stopping.
        save_path (str): Caminho para salvar o melhor modelo.

    Returns:
        nn.Module: Modelo treinado (melhor checkpoint).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        # Validação
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load(save_path))
    print("✅ Melhor modelo carregado.")
    return model
