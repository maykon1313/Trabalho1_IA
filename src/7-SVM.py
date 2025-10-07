from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from base import load, get_sentence_transformer, interactive_menu

def train_svm():
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    gamma_values = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    best_c = 0.1
    best_gamma = 0.001
    best_model = None
    best_f1_score = 0

    train_label, validation_label, test_label, train_embeddings_scaled, validation_embeddings_scaled, test_embeddings_scaled, scaler = load()

    for c in c_values:
        for g in gamma_values:
            svm = SVC(C=c, gamma=g, kernel='rbf')
            svm.fit(train_embeddings_scaled, train_label)

            val_predictions = svm.predict(validation_embeddings_scaled)
            val_accuracy = accuracy_score(validation_label, val_predictions)
            score = f1_score(validation_label, val_predictions, average='weighted')

            print(f"C: {c}, Gamma: {g}, Acurácia: {val_accuracy:.4f}, F1-score: {score:.4f}")
                
            if score > best_f1_score:
                best_c = c
                best_gamma = g
                best_model = svm
                best_f1_score = score

    if best_model is not None:
        # Fazer previsões no conjunto de teste
        test_predictions = best_model.predict(test_embeddings_scaled)
        test_accuracy = accuracy_score(test_label, test_predictions)
        test_f1 = f1_score(test_label, test_predictions, average='weighted')
        
        print(f"\nMelhor valor de C: {best_c}")
        print(f"Melhor valor de Gamma: {best_gamma}")
        print(f"F1-score na validação: {best_f1_score:.4f}")
        print(f"Acurácia no teste: {test_accuracy:.4f}")
        print(f"F1-score no teste: {test_f1:.4f}")

        # Relatório de classificação detalhado
        print("\nRelatório de Classificação no Conjunto de Teste:")
        print(classification_report(test_label, test_predictions))
    else:
        print("Erro: Nenhum modelo foi treinado com sucesso.")

    return best_c, best_gamma, best_model, scaler

def svm():
    print("treinando...")
    best_c, best_gamma, best_model, scaler = train_svm()
    
    if best_model is None:
        print("Erro ao carregar o modelo.")
        return

    transformer_model = get_sentence_transformer()

    info = f"Usando C = {best_c}, Gamma: {best_gamma}."
    
    interactive_menu(best_model, scaler, transformer_model, info)

if __name__ == "__main__":
    svm()
