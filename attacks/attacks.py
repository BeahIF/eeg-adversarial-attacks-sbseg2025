import torch
import numpy as np
import foolbox as fb

def generate_adversarial_examples(model, X, y, attack_name="FGSM", epsilon=0.1):
    """
    Gera exemplos adversariais usando Foolbox com o ataque especificado.

    Parâmetros:
        model: modelo PyTorch treinado
        X (np.ndarray): dados de entrada (shape: N, C, T)
        y (np.ndarray): rótulos verdadeiros
        attack_name (str): "FGSM", "PGD", "DeepFool" ou "CW"
        epsilon (float ou lista): valor ou lista de perturbações (somente FGSM suporta lista)

    Retorna:
        dict: chave é o epsilon (ou "default" para CW), valor é o X_adv correspondente
    """

    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=None)

    inputs = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.int64)

    adv_examples = {}

    if attack_name == "FGSM":
        attack = fb.attacks.FGSM()
        epsilons = [0.01, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.1]
        print("⚔️ Executando ataque FGSM com múltiplos epsilons...")

        for eps in epsilons:
            print(f"➡️ Epsilon = {eps}")
            _, clipped, _ = attack(fmodel, inputs, labels, epsilons=eps)
            adv_examples[eps] = clipped.cpu().numpy()

    elif attack_name == "PGD":
        attack = fb.attacks.LinfPGD()
        print(f"⚔️ Executando ataque PGD com epsilon={epsilon}...")
        _, clipped, _ = attack(fmodel, inputs, labels, epsilons=epsilon)
        adv_examples[epsilon] = clipped.cpu().numpy()

    elif attack_name == "DeepFool":
        attack = fb.attacks.LinfDeepFoolAttack()
        print(f"⚔️ Executando ataque DeepFool com epsilon={epsilon}...")
        _, clipped, _ = attack(fmodel, inputs, labels, epsilons=epsilon)
        adv_examples[epsilon] = clipped.cpu().numpy()

    elif attack_name == "CW":
        attack = fb.attacks.L2CarliniWagnerAttack(steps=100)
        print("⚔️ Executando ataque Carlini-Wagner (CW)...")
        _, clipped, _ = attack(fmodel, inputs, labels)
        adv_examples["default"] = clipped.cpu().numpy()

    else:
        raise ValueError(f"Ataque não suportado: {attack_name}")

    return adv_examples
