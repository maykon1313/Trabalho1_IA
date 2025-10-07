from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from base import load, get_sentence_transformer, interactive_menu

def train_lr():
    c_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    best_c = 0.01
    best_model = None
    best_f1_score = 0

    train_label, validation_label, test_label, train_embeddings_scaled, validation_embeddings_scaled, test_embeddings_scaled, scaler = load()

    for c in c_values:
        lr = LogisticRegression(C=c, max_iter=2000)
        lr.fit(train_embeddings_scaled, train_label)

        # Fazer previsões no conjunto de validação
        val_predictions = lr.predict(validation_embeddings_scaled)
        val_accuracy = accuracy_score(validation_label, val_predictions)
        score = f1_score(validation_label, val_predictions, average='weighted')

        print(f"C: {c}, Acurácia: {val_accuracy:.4f}, F1-score: {score:.4f}")

        if score > best_f1_score:
            best_c = c
            best_model = lr
            best_f1_score = score

    if best_model is not None:
        # Fazer previsões no conjunto de teste
        test_predictions = best_model.predict(test_embeddings_scaled)
        test_accuracy = accuracy_score(test_label, test_predictions)
        test_f1 = f1_score(test_label, test_predictions, average='weighted')

        print(f"\nMelhor valor de C: {best_c}")
        print(f"F1-score na validação: {best_f1_score:.4f}")
        print(f"Acurácia no teste: {test_accuracy:.4f}")
        print(f"F1-score no teste: {test_f1:.4f}")

        # Relatório de classificação detalhado
        print("\nRelatório de Classificação no Conjunto de Teste:")
        print(classification_report(test_label, test_predictions))
    else:
        print("Erro: Nenhum modelo foi treinado com sucesso.")

    return best_c, best_model, scaler

def lr():
    print("treinando...")
    best_c, best_model, scaler = train_lr()

    if best_model is None:
        print("Erro ao carregar o modelo.")
        return

    transformer_model = get_sentence_transformer()

    info = f"Usando C = {best_c}."

    interactive_menu(best_model, scaler, transformer_model, info)

if __name__ == "__main__":
    lr()
