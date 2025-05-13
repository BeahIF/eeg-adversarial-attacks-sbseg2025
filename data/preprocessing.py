import numpy as np
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import os


def load_and_save_bci_data(save_path_data="eeg_data.npy", save_path_labels="eeg_labels.npy"):
    """
    Carrega o dataset BCI IV 2a com MOABB, extrai os dados e salva em arquivos .npy

    Args:
        save_path_data (str): Caminho para salvar os dados EEG.
        save_path_labels (str): Caminho para salvar os rótulos.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, float]: (dados EEG, labels, frequência de amostragem)
    """
    dataset = BNCI2014_001()
    paradigm = MotorImagery()

    # Usar apenas um sujeito para experimentos rápidos (ou todos se preferir)
    subject_list = dataset.subject_list
    first_subject = subject_list[0]
    raw_data = dataset.get_data([first_subject])

    first_session_key = list(raw_data.keys())[0]
    first_run_key = list(raw_data[first_session_key].keys())[0]
    inner_key = list(raw_data[first_session_key][first_run_key].keys())[0]

    sfreq = raw_data[first_session_key][first_run_key][inner_key].info["sfreq"]

    X, labels, _ = paradigm.get_data(dataset)
    X = np.array(X, dtype=np.float32)
    labels = np.array(labels)

    np.save(save_path_data, X)
    np.save(save_path_labels, labels)

    print(f"Dados salvos: {X.shape}, Labels: {labels.shape}, Freq: {sfreq} Hz")
    return X, labels, sfreq


def normalize_data(X):
    """
    Normaliza os dados para o intervalo [0,1].

    Args:
        X (np.ndarray): Dados EEG.

    Returns:
        Tuple[np.ndarray, float, float]: Dados normalizados, mínimo, máximo.
    """
    X_min = X.min()
    X_max = X.max()
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized, X_min, X_max


if __name__ == "__main__":
    load_and_save_bci_data()