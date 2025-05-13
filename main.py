# main.py
import os
import numpy as np
import torch
from attacks.attacks import generate_adversarial_examples
from data.dataloader import get_dataloaders
from detector.detectors import run_detection_pipeline
from evaluate.metrics import evaluate_model
from models.eegnet import EEGNetv4

from data.preprocessing import load_and_save_bci_data, normalize_data
from training.train_model import train_model

def main():
    # Etapa 1: Baixar e processar os dados se ainda não existirem
    if not os.path.exists("eeg_data.npy") or not os.path.exists("eeg_labels.npy"):
        print("🔄 Baixando e processando dados...")
        load_and_save_bci_data()
        
       
    else:
        print("✅ Dados já existentes. Pulando download.")
    X = np.load("eeg_data.npy")
    y = np.load("eeg_labels.npy")
        
# 2. Pegar as informações para o modelo
    n_channels = X.shape[1]
    n_times = X.shape[2]
    n_classes = len(np.unique(y))
    # Etapa 2: Treinar o modelo
    print("🚀 Iniciando o treinamento...")
    model = EEGNetv4(in_chans=n_channels,
    n_classes=n_classes,
    input_window_samples=n_times)  
    X_normalized, _, _ = normalize_data(X)

    train_loader, val_loader = get_dataloaders(X_normalized, y)

    train_model(model, train_loader, val_loader)

    print("🎉 Treinamento finalizado!")

    model = torch.load("EEGNet_best_model.pth")
    print("✅ Melhor modelo carregado.")

    # ==========================
    # 🧪 Etapa 2: Avaliação normal
    # ==========================
    print("\n📊 Avaliando modelo no conjunto limpo...")
    model.load_state_dict(torch.load("EEGNet_best_model.pth"))

    evaluate_model(model, val_loader)

    # ==========================
    # ⚔️ Etapa 3: Ataque adversarial
    # ==========================
    attacks = ["FGSM", "PGD", "DeepFool", "CW"]
# Extrair X_val e y_val do val_loader (para usar nos ataques adversariais)
    X_val = []
    y_val = []

    for batch in val_loader:
        inputs, labels = batch
        X_val.append(inputs)
        y_val.append(labels)

    X_val = torch.cat(X_val, dim=0).numpy()
    y_val = torch.cat(y_val, dim=0).numpy()

    for attack_name in attacks:
        print(f"\n=== 🔥 Ataque: {attack_name} ===")

        if attack_name == "FGSM":
            # FGSM gera múltiplos epsilons
            adv_examples_dict = generate_adversarial_examples(
                model, X_val, y_val, attack_name="FGSM"
            )

            for eps, X_adv in adv_examples_dict.items():
                print(f"\n📊 Avaliação para FGSM com epsilon = {eps}")
                evaluate_model(model, X_adv, y_val, adversarial=True)

        elif attack_name in ["PGD", "DeepFool"]:
            # Definimos um epsilon padrão para PGD e DeepFool
            epsilon = 0.03
            adv_examples_dict = generate_adversarial_examples(
                model, X_val, y_val, attack_name=attack_name, epsilon=epsilon
            )

            for eps, X_adv in adv_examples_dict.items():
                print(f"\n📊 Avaliação para {attack_name} com epsilon = {eps}")
                evaluate_model(model, X_adv, y_val, adversarial=True)

        elif attack_name == "CW":
            adv_examples_dict = generate_adversarial_examples(
                model, X_val, y_val, attack_name="CW"
            )

            X_adv = adv_examples_dict["default"]
            print(f"\n📊 Avaliação para Carlini-Wagner")
            evaluate_model(model, X_adv, y_val, adversarial=True)

        print("\n✅ Ataques e avaliações concluídos.")

    # ==========================
    # 📊 Etapa 4: Avaliação adversarial
    # ==========================
    print("\n📊 Avaliação do modelo em dados adversariais...")
    evaluate_model(model, X_adv, y_val, adversarial=True)

    # ==========================
    # 🕵️ Etapa 5: Detector de ataques
    # ==========================
    print("\n🕵️ Rodando detector de exemplos adversariais...")
    run_detection_pipeline(X_normalized, X_adv)

    print("\n🎉 Pipeline completo executado com sucesso!")
if __name__ == "__main__":
    main()
