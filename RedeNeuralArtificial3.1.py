###############################################
# Rede Neural Artificial v3.1 - 2024          #
# By Vinicius Cunha Martins                   #
#                                             #
# Projeto de desenvolvimento de uma Rede      #
# Neural Perceptron Multicamada treinada      #
# com   Estratégias   Evolutivas    para      #
# problemas de classificação.                 #
###############################################
#EE-(µ+λ)

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, clear_output
from random import shuffle
import time, datetime
import copy
import csv


# Constantes
NUM_NEURONS = 4                         # Número de neurônios por camada escondida
NUM_OUTPUT_NEURONS = 4                  # Número de neurônios na camada de saída; quantidade de categorias para classificação
NUM_HIDDEN_LAYERS = 1                   # Número de camadas escondidas
NUM_GENES = 100                         # Número de indivíduos; tamanho das populações
NUM_CLONES = 10                         # Número de clones por indivíduo
NUM_EPOCHS = 101                        # Número de épocas de treinamento
TEST = 0.2                              # Proporção do conjuno de teste em relação ao dataset
NORMALIZE = True                        # Se "True", o dataset será normalizado


# Reading Dataset
with open('F:/Downloads/archive(2)/updated_pollution_dataset (2).csv', mode ='r') as file:
    csvfile = list(csv.reader(file))
    csvfile.pop(0)
    shuffle(csvfile)
    csvfile = np.array(csvfile)

    len_inputs = len(csvfile)

    csvfile = np.float64(csvfile)
    x = csvfile[:len_inputs, :-1]
    y = csvfile[:len_inputs, -1]

    file.close()


def normalize(x):
    # Calcula os mínimos e máximos de cada coluna
    min_vals = np.min(x, axis=0)
    max_vals = np.max(x, axis=0)

    diff = max_vals - min_vals
    diff[diff == 0] = 1
    # Normaliza os dados
    normalized_x = np.divide(x - min_vals, diff)

    return normalized_x

# Normalize Dataset
if NORMALIZE:
    x = normalize(x)

NUM_VARIABLES = len(x[0])+1      # 1 a mais para o viés

x = np.insert(x, len(x[0]), -1, axis=1)

# Train and Test split
x_test = x[int(len_inputs*(1 - TEST)):]
x_train = x[:int(len_inputs*(1 - TEST))]
y_test = y[int(len_inputs*(1 - TEST)):]
y_train = y[:int(len_inputs*(1 - TEST))]

print(x_train)
print(y_train)

def convert(n):   # Convert seconds to clock time
    return str(datetime.timedelta(seconds = n))

def sigmoid(x):   # Logistic Function
    return 1 / (1 + np.exp(-x*1))

def tanh(x):   # Hyperbolic Tangent Funtion
    return (np.exp(-1*x) - 1) / (np.exp(-1*x) + 1)

def softmax(values):   # Softmax Function
    return np.exp(values) / np.sum(np.exp(values))

def confusion_matrix(matrix):   # Config Confusion Matrix
    ax_.cla()

    # Calcular distâncias absolutas do valor fixo
    distancias = len(x_train) - matrix

    # Inverter para que valores mais próximos do valor fixo sejam mais escuros
    cores = len(x_train) - distancias

    # Usar o mapa de cores 'Blues' e aplicar a matriz de distâncias invertidas
    cax = ax_.matshow(cores, cmap=plt.cm.Blues, vmax=len(x_test)/NUM_OUTPUT_NEURONS, vmin=0)

    # Exibir valores na matriz
    for i in range(NUM_OUTPUT_NEURONS):
        for j in range(NUM_OUTPUT_NEURONS):
            c = matrix[j, i]
            ax_.text(i, j, str(c), va='center', ha='center', color='black')

    return cax

def initialize_generation():
    generations = []
    for i in range(NUM_GENES):
        gene = []
        # Primeira camada
        gene.append(np.random.uniform(-1, 1, (NUM_NEURONS, NUM_VARIABLES)))
        # Camadas ocultas
        for j in range(NUM_HIDDEN_LAYERS - 1):
            gene.append(np.random.uniform(-1, 1, (NUM_NEURONS, NUM_NEURONS+1)))
        # Camada de saída
        gene.append(np.random.uniform(-1, 1, (NUM_OUTPUT_NEURONS, NUM_NEURONS)))

        generations.append(gene)
    return generations

def cloning(generations):
    return [copy.deepcopy(gene) for gene in generations for _ in range(NUM_CLONES)]

def mutate_weights(weights):
    return weights + np.random.uniform(-noise_magnitude, noise_magnitude, weights.shape) / 100

def mutation(clones):
    mutations = copy.deepcopy(clones)
    for h in range(NUM_GENES):
        for i in range(NUM_CLONES - 1):  # Deixa o último clone sem mutação; Mantém original
            for j in range(NUM_HIDDEN_LAYERS + 1):
                mutations[h * NUM_CLONES + i][j] = mutate_weights(mutations[h * NUM_CLONES + i][j])
    return mutations

def out_layer(mutations):
    scores = np.zeros((NUM_GENES, NUM_CLONES))
    new_generation = []

    for g, variables in enumerate(x_train):
        for h in range(NUM_GENES):
            for i in range(NUM_CLONES):
                values = np.zeros((NUM_HIDDEN_LAYERS + 1, NUM_NEURONS))

                # Primeira camada
                values[0] = sigmoid(np.dot(mutations[h * NUM_CLONES + i][0], variables))   #np.maximum(0, np.dot(mutations[h * NUM_CLONES + i][0], variables))   #tanh(np.dot(mutations[h * NUM_CLONES + i][0], variables))   #

                # Camadas ocultas
                for j in range(1, NUM_HIDDEN_LAYERS):
                    values[j] = np.maximum(0, np.dot(mutations[h * NUM_CLONES + i][j], np.append(values[j-1], -1)))   #tanh(np.dot(mutations[h * NUM_CLONES + i][j], values[j-1]))   #sigmoid(np.dot(mutations[h * NUM_CLONES + i][j], values[j-1]))   #

                # Camada de saída
                final_output = softmax(np.dot(mutations[h * NUM_CLONES + i][-1], values[-2]))
                prediction = np.argmax(final_output)

                if prediction == y_train[g]:
                    scores[h][i] += 1 #(final_output[prediction])**(1/4)  # Pontua os clones de cada indivíduo
                #else:
                #    scores[h][i] -= (final_output[prediction])**4# + 0.1

    best_indices = np.argmax(scores, axis=1)
    for h in range(NUM_GENES):
        new_generation.append(mutations[h * NUM_CLONES + best_indices[h]])

    return new_generation, scores

def test(mutations, input, output):
    scores = np.zeros(NUM_GENES)
    matrix = np.zeros((NUM_GENES, NUM_OUTPUT_NEURONS, NUM_OUTPUT_NEURONS), dtype=int)  #Confusion Matrix

    for g, variables in enumerate(input):
        for h in range(NUM_GENES):
            values = np.zeros((NUM_HIDDEN_LAYERS + 1, NUM_NEURONS))

            # Primeira camada
            values[0] = sigmoid(np.dot(mutations[h][0], variables))   #np.maximum(0, np.dot(mutations[h][0], variables))  #tanh(np.dot(mutations[h][0], variables))   #

            # Camadas ocultas
            for j in range(1, NUM_HIDDEN_LAYERS):
                values[j] = np.maximum(0, np.dot(mutations[h][j], np.append(values[j-1], -1)))  #tanh(np.dot(mutations[h][j], values[j-1]))   #sigmoid(np.dot(mutations[h][j], values[j-1]))   #

            # Camada de saída
            final_output = softmax(np.dot(mutations[h][-1], values[-2]))
            prediction = np.argmax(final_output)

            if prediction == output[g]:
                scores[h] += 1 #(final_output[prediction])**(1/4)   # Pontua o indivíduo
            #else:
            #    scores[h] -= (final_output[prediction])**4# + 0.1

            matrix[h][int(output[g])][int(prediction)] += 1


    return scores, matrix[np.argmax(scores)]

# Repeat training
for i in range(3):
    # Configurações da plotagem
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (4,3)
    fig_, ax_ = plt.subplots()
    plt.rcParams['figure.figsize'] = (8,6)
    fig, ax = plt.subplots()
    ax.set_title('Taxa de Erro por Época')
    ax.set_xlabel('Época')
    ax.set_ylabel('Taxa de Erro (%)')
    ax.grid(True)
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()

    noise_magnitude = 30
    # Inicializa a primeira geração
    generation = initialize_generation()

    # Loop principal para evoluir as gerações
    clone_mean = []
    best_scores = []
    test_mean = []
    test_best = []

    for epoch in range(NUM_EPOCHS):
        t_atual = time.time()

        clones = cloning(generation)
        mutations = mutation(clones)
        generation, score = out_layer(mutations)
        score_test, matrix_test = test(generation, x_test, y_test)
        score_train, matrix_train = test(generation, x_train, y_train)

        clone_mean.append((len(x_train) - (np.mean(np.max(score, axis=1)))) * 100/len(x_train))
        best_scores.append((len(x_train) - np.max(np.max(score, axis=1))) * 100/len(x_train))
        test_mean.append((len(x_test) - np.mean(score_test)) * 100 / len(x_test))
        test_best.append((len(x_test) - np.max(score_test)) * 100 / len(x_test))


        t_estimado = convert(int((time.time() - t_atual) * (NUM_EPOCHS - len(clone_mean))))
        print(f'{round(len(clone_mean)/NUM_EPOCHS * 100, 2)}%', f'Tempo restante estimado: {t_estimado}')
        #print(np.mean(np.max(score, axis=1)), np.max(np.max(score, axis=1)))

        ax.set_ylim(0, 100)
        ax.plot(range(len(clone_mean)), clone_mean, label='Média', color='blue', linewidth=0.5)
        ax.plot(range(len(best_scores)), best_scores, label='Melhor indivíduo', color='purple', linewidth=0.5)
        ax.plot(range(len(test_mean)), test_mean, label='Média (Teste)', color='red', linewidth=0.5)
        ax.plot(range(len(test_best)), test_best, label='M. indiv. (Teste)', color='orange', linewidth=0.5)

        cax = confusion_matrix(matrix_test)

        if len(clone_mean) == 1:
            ax.legend()
            # Exibir gráfico
            plt.colorbar(cax)  # Adicionar barra de cores para referência
        if len(clone_mean) % 3 == 0:
            noise_magnitude *= 0.90
            #print(noise_magnitude)


        sum = 0
        for j in range(NUM_OUTPUT_NEURONS):
            sum += matrix_train[j][j]
        accuracy_train = round((100 * sum)/(np.sum(np.sum(matrix_train, axis=1))), 2)
        sum = 0
        for j in range(NUM_OUTPUT_NEURONS):
            sum += matrix_test[j][j]
        accuracy_test = round((100 * sum)/(np.sum(np.sum(matrix_test, axis=1))), 2)

        print("                         ", np.argmax(np.max(score, axis=1)), "    ", np.argmax(score_test))
        print(f"Acurácia (treino, teste): {accuracy_train}%, {accuracy_test}%\n\n")


        clear_output(wait=True)
        display(fig)
        plt.pause(2)

        clear_output(wait=True)
        display(fig_)
        plt.pause(2)


        if epoch == NUM_EPOCHS - 1:
            time.sleep(1)
            print('Salvando gráfico...')
            plt.savefig(f'F:/Downloads/Estudo_RNA_heart_disease/{i}_grafico_{NUM_NEURONS}n_{NUM_HIDDEN_LAYERS}hl.png')
            plt.close()

            time.sleep(1)
            print('Salvando matriz de confusão...')
            plt.savefig(f'F:/Downloads/Estudo_RNA_heart_disease/{i}_matriz_de_confusao_{NUM_NEURONS}n_{NUM_HIDDEN_LAYERS}hl.png')
            plt.close()
