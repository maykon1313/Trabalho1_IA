import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

def load():
# Carregar os embeddings e labels salvos
    data = np.load('data/feiticos_embeddings.npz')

    train_embeddings = data['train_embeddings']
    validation_embeddings = data['validation_embeddings']
    test_embeddings = data['test_embeddings']

    train_label = data['train_label']
    validation_label = data['validation_label']
    test_label = data['test_label']

    # Normalizar os dados para melhorar a performance do KNN
    scaler = StandardScaler()

    train_embeddings_scaled = scaler.fit_transform(train_embeddings)
    validation_embeddings_scaled = scaler.transform(validation_embeddings)
    test_embeddings_scaled = scaler.transform(test_embeddings)

    return train_label, validation_label, test_label, train_embeddings_scaled, validation_embeddings_scaled, test_embeddings_scaled, scaler

def train():
    # Valores de K e métricas de distância
    k_values = list(range(1, 31)) 
    metrics = ['euclidean', 'manhattan', 'cosine', 'minkowski', 'hamming']

    best_k = 1
    best_model = None
    best_metric = 'euclidean'
    best_accuracy = 0

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
            print(f"K: {k}, Métrica: {metric}, Acurácia: {val_accuracy:.4f}")
            
            if val_accuracy > best_accuracy:
                best_k = k
                best_model = knn
                best_metric = metric
                best_accuracy = val_accuracy

    if best_model is not None:
        # Fazer previsões no conjunto de teste
        test_predictions = best_model.predict(test_embeddings_scaled)
        test_accuracy = accuracy_score(test_label, test_predictions)
        print(f"\nMelhor valor de K: {best_k}")
        print(f"Melhor métrica: {best_metric}")
        print(f"Acurácia de validação: {best_accuracy:.4f}")
        print(f"Acurácia no teste: {test_accuracy:.4f}")

        # Relatório de classificação detalhado
        print("\nRelatório de Classificação no Conjunto de Teste:")
        print(classification_report(test_label, test_predictions))
    else:
        print("Erro: Nenhum modelo foi treinado com sucesso.")

    return best_k, best_model, best_metric, scaler

def main():
    print("treinando...")
    best_k, best_model, best_metric, scaler = train()
    
    while True:
        print("Deseja testar para um input personalizado? (s/n)")
        resp = input()

        if resp == "n":
            print("Saindo.")
            return

        elif resp == "s" and best_model != None:
            print("Digite a escola: ")
            school = input()

            print("Digite a descrição do feitiço: ")
            sentence = input()

            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            sen_embed = model.encode(sentence, convert_to_numpy=True)

            sen_embed_scaled = scaler.transform([sen_embed])

            prediction = best_model.predict(sen_embed_scaled)

            print(f"Usando K = {best_k} e Metrica = {best_metric}.")

            if prediction[0] == school:
                print(f"O modelo corretamente acertou a escola: {prediction[0]}.")
            
            else:
                print(f"O modelo não acertou a escola: {prediction[0]}.")

        else:
            print("Erro ao carregar o modelo ou input.")

if __name__ == "__main__":
    main()