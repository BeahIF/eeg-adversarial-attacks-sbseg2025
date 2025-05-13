import torch.nn as nn
from braindecode.models import EEGNetv4

def create_eegnet_model(n_channels, n_times, n_classes, sfreq):
    """
    Cria uma instância do modelo EEGNetv4 com os parâmetros especificados.

    Args:
        n_channels (int): Número de canais de EEG.
        n_times (int): Número de amostras temporais.
        n_classes (int): Número de classes alvo.
        sfreq (float): Frequência de amostragem.

    Returns:
        EEGNetv4: Modelo EEGNetv4 instanciado.
    """
    model = EEGNetv4(
        in_chans=n_channels,
        n_classes=n_classes,
        input_window_seconds=None,
        n_times=n_times,
        sfreq=sfreq
    )
    return model
