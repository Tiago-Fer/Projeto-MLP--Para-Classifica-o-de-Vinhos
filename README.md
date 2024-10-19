Projeto MLP - Multi-Layer Perceptron para Classificação de Vinhos
Este repositório contém o código e os detalhes do projeto desenvolvido como parte de um projeto extra-acadêmico voltado para a implementação de uma Rede Neural Perceptron Multicamadas (MLP). O objetivo do projeto é a classificação da qualidade de vinhos utilizando características químicas dos vinhos.

Estrutura do Projeto
Código principal: O código da rede neural está implementado no arquivo RedeNeuralMLP.py.
Relatório: O relatório completo detalhando a implementação e os desafios pode ser encontrado no arquivo Relatorio Multi-Layer Perceptron- MLP.docx.
Descrição do Projeto
O objetivo deste projeto é criar e treinar um modelo de rede neural MLP para prever a qualidade de vinhos com base em várias características químicas, como acidez, teor de açúcar residual, pH, e teor alcoólico.

A rede neural foi desenvolvida utilizando as bibliotecas TensorFlow e Keras e a técnica de oversampling SMOTE para balancear o dataset. A arquitetura da MLP consiste em:

Camada de entrada: Número de neurônios igual ao número de características de entrada (11 características).
Duas camadas ocultas: 95 e 40 neurônios, utilizando a função de ativação ReLU.
Camada de saída: 6 neurônios para classificação das classes de qualidade (ajustadas para 6 classes), utilizando a função de ativação Softmax.
Dependências
As bibliotecas e ferramentas usadas no projeto incluem:

Pandas
NumPy
TensorFlow
Keras
imbalanced-learn (para a técnica de oversampling SMOTE)
scikit-learn
Você pode instalar as dependências executando:

bash
Copiar código
pip install -r requirements.txt
Estrutura da Rede Neural
A rede neural foi configurada com inicialização de pesos utilizando glorot_uniform e biases inicializados com zero.
O otimizador Adam foi utilizado para treinar a rede neural, com a função de perda categorical_crossentropy para problemas de classificação multiclasse.
Foi implementado o Early Stopping para interromper o treinamento quando a performance no conjunto de validação parasse de melhorar.
Instruções de Execução
Clone o repositório:
bash
Copiar código
git clone https://github.com/seuusuario/projeto-mlp-classificacao-vinhos.git
Navegue até o diretório do projeto:
bash
Copiar código
cd projeto-mlp-classificacao-vinhos
Execute o script principal RedeNeuralMLP.py:
bash
Copiar código
python RedeNeuralMLP.py
O script treinará o modelo e exibirá os resultados de avaliação, incluindo a acurácia e o relatório de classificação.

Dataset
O dataset utilizado foi pré-processado utilizando a técnica SMOTE para balanceamento das classes, e foi dividido em conjuntos de treino e teste.

Os datasets podem ser carregados diretamente pelo código, desde que estejam no mesmo diretório que o script principal:

UNIAO_DFS_TREINO.csv - Dataset de treino.
dataset_PRIMEIRO_TESTE.csv e dataset_TERCEIRO_TESTE.csv - Datasets de teste.
Resultados
A MLP alcançou uma acurácia de aproximadamente 60% no conjunto de teste, com uma perda de cerca de 1.45. A performance do modelo foi fortemente influenciada pelos ajustes de hiperparâmetros, como taxa de aprendizado e batch size, além do uso de regularização L2 para evitar overfitting.

Contribuição
Sinta-se à vontade para abrir issues ou pull requests caso deseje contribuir para este projeto.

Agradecimentos
Gostaria de agradecer às orientadoras, Prof.ª Vanessa Matias Leite e Prof.ª Elisa Antolli, por todo o apoio e orientação durante o desenvolvimento deste projeto, e ao meu colega Filipe Ambrozio pelas discussões e contribuições ao código.