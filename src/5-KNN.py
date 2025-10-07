from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from base import load, get_sentence_transformer, interactive_menu

def train_knn():
    # Valores de K e métricas de distância
    k_values = list(range(1, 31, 2)) 
    metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski', 'hamming']

    best_k = 1
    best_model = None
    best_metric = 'euclidean'
    best_f1_score = 0

    train_label, validation_label, test_label, train_embeddings_scaled, validation_embeddings_scaled, test_embeddings_scaled, scaler = load()

    # Criar e treinar o modelo KNN
    for k in k_values:
        # Testar diferentes métricas de distância
        for metric in metrics:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights='distance')
            knn.fit(train_embeddings_scaled, train_label)

            # Fazer previsões no conjunto de validação
            val_predictions = knn.predict(validation_embeddings_scaled)
            val_accuracy = accuracy_score(validation_label, val_predictions)
            score = f1_score(validation_label, val_predictions, average='weighted')

            print(f"K: {k}, Métrica: {metric}, Acurácia: {val_accuracy:.4f}, F1-score: {score:.4f}")

            if score > best_f1_score:
                best_k = k
                best_metric = metric
                best_model = knn
                best_f1_score = score

    if best_model is not None:
        # Fazer previsões no conjunto de teste
        test_predictions = best_model.predict(test_embeddings_scaled)
        test_accuracy = accuracy_score(test_label, test_predictions)
        test_f1 = f1_score(test_label, test_predictions, average='weighted')

        print(f"\nMelhor valor de K: {best_k}")
        print(f"Melhor métrica: {best_metric}")
        print(f"F1-score na validação: {best_f1_score:.4f}")
        print(f"Acurácia no teste: {test_accuracy:.4f}")
        print(f"F1-score no teste: {test_f1:.4f}")

        # Relatório de classificação detalhado
        print("\nRelatório de Classificação no Conjunto de Teste:")
        print(classification_report(test_label, test_predictions))
    else:
        print("Erro: Nenhum modelo foi treinado com sucesso.")

    return best_k, best_metric, best_model, scaler

def knn():
    print("treinando...")
    best_k, best_metric, best_model, scaler = train_knn()

    if best_model is None:
        print("Erro ao carregar o modelo.")
        return

    transformer_model = get_sentence_transformer()

    info = f"Usando K = {best_k} e Metrica = {best_metric}."
    
    interactive_menu(best_model, scaler, transformer_model, info)

if __name__ == "__main__":
    knn()
