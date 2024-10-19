import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# Criando a classe MLP, que será responsável por definir a rede neural e preparar os dados.
class RedeMLP:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))
        model.add(Dense(95, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(0.004)))
        model.add(Dense(40, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(0.004)))
        model.add(Dense(6, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=l2(0.004)))
        return model
    
    def compilador(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def treino(self, X_train, y_train, epochs=128, batch_size=8, validation_data=None, callbacks=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1)

# Carregando o dataset UNIAO_DFS_TREINO.csv
data_treino = pd.read_csv('UNIAO_DFS_TREINO.csv', sep=';')

# Aplicando SMOTE para aumentar o dataset
X = data_treino.drop('quality', axis=1).values
y = data_treino['quality'].values

smote = SMOTE(random_state=212, k_neighbors=2)
X_smote, y_smote = smote.fit_resample(X, y)

data_treino_smote = pd.DataFrame(X_smote, columns=data_treino.drop('quality', axis=1).columns)
data_treino_smote['quality'] = y_smote

data_treino_smote = data_treino_smote[data_treino_smote['quality'].between(3, 8)]

# Carregando os datasets de Teste
data_teste = pd.read_csv('dataset_PRIMEIRO_TESTE.csv', sep=';')
data_terceiro_teste = pd.read_csv('dataset_TERCEIRO_TESTE.csv', sep=';')

data_teste = pd.concat([data_teste, data_terceiro_teste])
data_teste = data_teste[data_teste['quality'].between(3, 8)]

# Ajuste das casas decimais.
data_treino_smote = data_treino_smote.round({'fixed acidity': 1, 'volatile acidity': 3, 'citric acid': 2, 
                                             'residual sugar': 2, 'chlorides': 3, 'free sulfur dioxide': 1, 
                                             'total sulfur dioxide': 1, 'density': 5, 'pH': 2, 'sulphates': 2, 
                                             'alcohol': 1})

data_teste = data_teste.round({'fixed acidity': 1, 'volatile acidity': 3, 'citric acid': 2, 
                               'residual sugar': 2, 'chlorides': 3, 'free sulfur dioxide': 1, 
                               'total sulfur dioxide': 1, 'density': 5, 'pH': 2, 'sulphates': 2, 
                               'alcohol': 1})

# Separando as features e labels.
X_train = data_treino_smote.drop('quality', axis=1).values
y_train = pd.get_dummies(data_treino_smote['quality'] - 3).values

X_test = data_teste.drop('quality', axis=1).values

y_test = np.zeros((data_teste.shape[0], 6))
for idx, val in enumerate(data_teste['quality']):
    y_test[idx, val - 3] = 1

# Normalização
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Verificação
print(f'Shape de X_train: {X_train.shape}')
print(f'Shape de y_train (one-hot): {y_train.shape}')
print(f'Shape de X_test: {X_test.shape}')
print(f'Shape de y_test (one-hot): {y_test.shape}')

# Instanciando a rede neural.
mlp = RedeMLP(input_shape=X_train.shape[1])

# Compilando o modelo.
mlp.compilador()

# Treinamento com Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp.treino(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Avaliação do modelo
score = mlp.evaluate(X_test, y_test)
print(f'\nLoss no conjunto de teste: {score[0]}')
print(f'Acuracia no conjunto de teste: {score[1]}')

# Predição e relatório de classificação
y_pred = mlp.predict(X_test)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
