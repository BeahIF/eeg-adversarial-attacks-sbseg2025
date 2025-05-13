import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def run_detection_pipeline(X_clean, X_adv):
    """
    Treina e avalia detectores de anomalias (RandomForest, SVM, KNN)
    com dados limpos e adversariais misturados.
    """

    print("🕵️ Treinando detectores com RandomForest, SVM e KNN...")

    # Combina dados
    X_combined = np.concatenate([X_clean, X_adv])
    y_combined = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_adv))])  # 0 = limpo, 1 = adversarial

    # Extrair features simples: média por canal
    X_features = X_combined.mean(axis=2)  # shape: (amostras, canais)

    # Modelos de detecção
    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    for model_name, model in models.items():
        print(f"\n🔍 Treinando detector: {model_name}...")

        # Treinar o modelo
        model.fit(X_features, y_combined)

        # Fazer previsões
        preds = model.predict(X_features)

        # Avaliação do modelo
        print(f"\n📊 Relatório do detector {model_name}:")
        print(classification_report(y_combined, preds, target_names=["Clean", "Adversarial"]))

    print("\n✅ Todos os detectores foram avaliados.")
