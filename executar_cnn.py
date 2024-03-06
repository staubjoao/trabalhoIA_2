from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import KFold
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import pickle

input_shape = (256, 256, 3)
classes = ["beach", "bus", "cafe_restaurant", "car", "city_center", "forest_path", "grocery_store",
           "home", "library", "metro_station", "office", "park", "residential_area", "train", "tram"]


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, fold=0):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.0f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')

    plt.savefig("fusion2/confusion_matrix_fold_"+str(fold)+".png")

    plt.close()


def cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               input_shape=input_shape, padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


labels = []
images = []

# Leitura dos dados para a CNN
path = './spec_evaluation_janela/'
max_imagens = 1170
j = 0
k = 0
for i in range(1170):
    aux = []
    if j == 78:
        j = 0
        k += 1
    path_total = path + 'class_' + classes[k] + '/' + str(i) + '/'
    for l in range(6):
        path_img = path_total + str(l) + '.png'
        img = cv2.imread(path_img)
        img = cv2.resize(img, (256, 256))
        aux.append(img)
    images.append(aux)
    labels.append(k)
    j += 1

histories = []
scores_array = []
y_preds = []


# Definição dos valores para treinamento
k_fold = 5
epochs = 1
batch_size = 64

# Para a CNN
acc_per_fold = []
loss_per_fold = []
best_model, best_acc = None, 0.0

# Split dos folds
kfold = KFold(n_splits=k_fold, shuffle=True)

# For para treinamento entre os folds
fold_no = 1
for train, test in kfold.split(images, labels):
    # Criação do modelo
    model = cnn_model()

    print('------------------------------------------------------------------------')
    print(f'Treinamento para o fold {fold_no} ...')

    # Vetores para teste e treinamento
    x_train_cnn = []
    y_train_cnn = []

    x_test_cnn = []
    y_test_cnn = []

    # Fazendo o split dos dados para treinamento
    for i in train:
        for j in range(6):
            x_train_cnn.append(images[i][j])
            y_train_cnn.append(labels[i])

    # Fazendo o split dos dados para teste
    for i in test:
        for j in range(6):
            x_test_cnn.append(images[i][j])
            y_test_cnn.append(labels[i])

    # Tranformando em vetores numpy
    x_train_cnn = np.array(x_train_cnn)
    x_test_cnn = np.array(x_test_cnn)
    y_train_cnn = np.array(y_train_cnn)
    y_test_cnn = np.array(y_test_cnn)

    y_pred = y_test_cnn[:]
    for res in y_pred:
        y_preds.append(res)

    y_train_cnn = to_categorical(y_train_cnn, num_classes=len(classes))
    y_test_cnn = to_categorical(y_test_cnn, num_classes=len(classes))

    # CNN
    history = model.fit(x_train_cnn, y_train_cnn,
                        batch_size=batch_size,
                        epochs=epochs)
    histories.append(history.history)
    scores = model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
    cnn_predictions = model.predict(x_test_cnn)

    # Salvar predições em texto
    np.savetxt(f'predicoes_final/predicoes_real_fold{fold_no}.txt',
               y_pred, fmt="%f", delimiter=';')
    np.savetxt(f'predicoes_final/predicoes_cnn_fold{fold_no}.txt',
               cnn_predictions, fmt="%f", delimiter=';')

    # Recolhe as melhores accuracias no modelo CNN para salvar depois
    if scores[1] > best_acc:
        best_acc = scores[1]
        best_model = model
    scores_array.append(scores)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    fold_no += 1

loss = []
accuracy = []

# Para plotar os graficos para CNN
for i in range(k_fold):
    aux_loss = histories[i]['loss']
    aux_accuracy = histories[i]['accuracy']

    loss.append(aux_loss)
    accuracy.append(aux_accuracy)

# Plot dos graficos da CNN
plt.figure(figsize=(15, 5))
plt_loss = plt.subplot(121)
for fold in range(len(loss)):
    plt.plot(loss[fold], label=f'fold {fold+1}')
plt.title("Perda")
plt.ylabel("Perda")
plt.xlabel("Época")
plt.legend()

plt_accuracy = plt.subplot(122)
for fold in range(len(accuracy)):
    plt.plot(accuracy[fold], label=f'fold {fold+1}')
plt.title("Acurácia")
plt.ylabel("Acurácia")
plt.xlabel("Época")
plt.legend()

# Salvar os gráficos em formato PNG
plt.savefig("saida_final/graficos.png")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.boxplot(loss)
plt.title('Validação Perda')
plt.xlabel('fold')
plt.ylabel('Perda')

plt.subplot(1, 2, 2)
plt.boxplot(accuracy)
plt.title('Validação Acurácia')
plt.xlabel('fold')
plt.ylabel('Acurácia')

plt.tight_layout()
plt.savefig("saida_final/boxplot.png")

# Salvar modelo
model_json = model.to_json()
with open("saida_final/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("saida_final/model.h5")
print("Saved model to disk")
