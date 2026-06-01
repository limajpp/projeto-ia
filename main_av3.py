import time
import numpy as np

from utils.loader import load_local_arff
from utils.splitter import k_fold_split
from utils.encoder import label_encode_column
from utils.metrics import classification_metrics, mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score, adjusted_r2_score
from models.perceptron import OvAPerceptron
from models.mlp import MLPRegressor

def run_classification_experiments():
    print("="*50)
    print(" EXPERIMENTOS DE CLASSIFICAÇÃO - PERCEPTON SIMPLES")
    print("="*50)
    
    loaded = load_local_arff(r'data/dataset_ STUDENTS_DROPOUT_AND_ACADEMIC_SUCCESS') 
    dataset = np.array(loaded[0] if isinstance(loaded, tuple) else loaded)
    
    X_raw = dataset[:, :-1]
    y_raw = dataset[:, -1]
    
    encode_result = label_encode_column(y_raw)
    y_encoded = encode_result[0] if isinstance(encode_result, tuple) else encode_result
    y_encoded = np.array(y_encoded).flatten().astype(int)
    
    X_numeric = np.zeros(X_raw.shape)
    for i in range(X_raw.shape[1]):
        try:
            X_numeric[:, i] = X_raw[:, i].astype(float)
        except ValueError:
            _, encoded = np.unique(X_raw[:, i], return_inverse=True)
            X_numeric[:, i] = encoded
    X = X_numeric
    
    learning_rates = [0.1, 0.01]
    epochs_list = [100, 500]
    
    results = []

    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"\n[Config] LR: {lr} | Épocas: {epochs}")
            
            acc_list, prec_list, rec_list, spec_list, f1_list = [], [], [], [], []
            last_confusion_matrix = None
            start_time = time.time()
            
            folds = k_fold_split(X, k=5)
            
            for train_idx, test_idx in folds:
                X_train, y_train = X[train_idx], y_encoded[train_idx]
                X_test, y_test = X[test_idx], y_encoded[test_idx]
                
                model = OvAPerceptron(learning_rate=lr, epochs=epochs)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                metrics = classification_metrics(y_test, y_pred)
                acc_list.append(metrics['accuracy'])
                prec_list.append(metrics['precision'])
                rec_list.append(metrics['recall'])
                spec_list.append(metrics['specificity'])
                f1_list.append(metrics['f1_score'])
                last_confusion_matrix = metrics['confusion_matrix']
                
            end_time = time.time()
            
            print(f"Tempo Total (5 Folds): {end_time - start_time:.4f}s")
            print(f"Acurácia: {np.mean(acc_list):.4f} | Precisão: {np.mean(prec_list):.4f}")
            print(f"Recall: {np.mean(rec_list):.4f} | Especificidade: {np.mean(spec_list):.4f}")
            print(f"F1-Score: {np.mean(f1_list):.4f}")
            print(f"Matriz de Confusão (último fold):\n{last_confusion_matrix}")

            results.append({
                'lr': lr, 'epochs': epochs,
                'accuracy': round(float(np.mean(acc_list)), 4),
                'precision': round(float(np.mean(prec_list)), 4),
                'recall': round(float(np.mean(rec_list)), 4),
                'specificity': round(float(np.mean(spec_list)), 4),
                'f1': round(float(np.mean(f1_list)), 4),
                'time': round(end_time - start_time, 4),
                'confusion_matrix': last_confusion_matrix.tolist() if last_confusion_matrix is not None else None,
            })

    return results


def run_regression_experiments():
    print("\n" + "="*50)
    print(" EXPERIMENTOS DE REGRESSÃO - MULTI-LAYER PERCEPTRON")
    print("="*50)
    
    loaded = load_local_arff(r'data/dataset_ANIME')
    dataset = np.array(loaded[0] if isinstance(loaded, tuple) else loaded)
    
    X_raw = dataset[:, :-2]
    y_raw = dataset[:, -2]
    
    X_numeric = np.zeros(X_raw.shape)
    for i in range(X_raw.shape[1]):
        try:
            X_numeric[:, i] = X_raw[:, i].astype(float)
        except ValueError:
            _, encoded = np.unique(X_raw[:, i].astype(str), return_inverse=True)
            X_numeric[:, i] = encoded
    X = X_numeric
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1e-8 
    X = (X - X_mean) / X_std
    
    y = np.array(y_raw).flatten().astype(float)
    
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    X = np.nan_to_num(X, nan=0.0)
    
    num_features = X.shape[1]

    topologies        = [(5,), (10,), (10, 5)]
    activations       = ['relu', 'sigmoid']
    learning_rates    = [0.001, 0.01]
    epochs_list       = [500, 1000]

    topologies_ext    = [(8,), (16,), (8, 4), (20, 10)]
    activations_ext   = ['sigmoid']          
    learning_rates_ext = [0.005, 0.05]       
    epochs_list_ext   = [2000, 5000]         

    results = []

    def run_config(top, act, lr, epochs, rodada):
        print(f"\n[{rodada}] Ocultas: {top} | Ativação: {act.upper()} | LR: {lr} | Épocas: {epochs}")
        mse_list, rmse_list, mae_list, r2_list, r2adj_list = [], [], [], [], []
        train_times, test_times = [], []
        folds = k_fold_split(X, k=5)
        for train_idx, test_idx in folds:
            X_train, y_train = X[train_idx], y[train_idx]
            X_test,  y_test  = X[test_idx],  y[test_idx]
            model = MLPRegressor(hidden_layers=top, activation=act, learning_rate=lr, epochs=epochs)
            t0 = time.time(); model.fit(X_train, y_train); train_times.append(time.time() - t0)
            t0 = time.time(); y_pred = model.predict(X_test); test_times.append(time.time() - t0)
            mse_list.append(mean_squared_error(y_test, y_pred))
            rmse_list.append(root_mean_squared_error(y_test, y_pred))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))
            r2adj_list.append(adjusted_r2_score(y_test, y_pred, num_features))
        print(f"Tempo Treino: {np.mean(train_times):.4f}s/fold | Tempo Teste: {np.mean(test_times):.6f}s/fold")
        print(f"MSE: {np.mean(mse_list):.4f} | RMSE: {np.mean(rmse_list):.4f} | MAE: {np.mean(mae_list):.4f}")
        print(f"R2: {np.mean(r2_list):.4f} | R2 Ajustado: {np.mean(r2adj_list):.4f}")
        return {
            'rodada': rodada,
            'topology': str(top), 'activation': act.upper(), 'lr': lr, 'epochs': epochs,
            'mse':     round(float(np.mean(mse_list)), 4),
            'rmse':    round(float(np.mean(rmse_list)), 4),
            'mae':     round(float(np.mean(mae_list)), 4),
            'r2':      round(float(np.mean(r2_list)), 4),
            'r2_adj':  round(float(np.mean(r2adj_list)), 4),
            'train_time': round(float(np.mean(train_times)), 4),
            'test_time':  round(float(np.mean(test_times)), 6),
        }

    print("\n" + "-"*50)
    print(" RODADA 1 — Configurações Originais")
    print("-"*50)
    for top in topologies:
        for act in activations:
            for lr in learning_rates:
                for epochs in epochs_list:
                    results.append(run_config(top, act, lr, epochs, 'Rodada 1'))

    print("\n" + "-"*50)
    print(" RODADA 2 — Configurações Estendidas")
    print("-"*50)

    for top in topologies:
        for act in activations_ext:
            for lr in learning_rates_ext:
                for epochs in epochs_list_ext:
                    results.append(run_config(top, act, lr, epochs, 'Rodada 2'))

    for top in topologies_ext:
        for act in activations_ext:
            for lr in [0.01] + learning_rates_ext:
                for epochs in [1000] + epochs_list_ext:
                    results.append(run_config(top, act, lr, epochs, 'Rodada 2'))

    return results


if __name__ == "__main__":
    clf_results = run_classification_experiments()
    reg_results = run_regression_experiments()

    import json, os
    os.makedirs('results', exist_ok=True)
    with open('results/clf_results.json', 'w') as f:
        json.dump(clf_results, f, indent=2)
    with open('results/reg_results.json', 'w') as f:
        json.dump(reg_results, f, indent=2)
    print("\n✓ Resultados salvos em results/clf_results.json e results/reg_results.json")

