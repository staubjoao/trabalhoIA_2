import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import pickle
import gc


def main():
    entrada = "./meta.txt"

    classes = ["beach", "bus", "cafe_restaurant", "car", "city_center", "forest_path", "grocery_store",
               "home", "library", "metro_station", "office", "park", "residential_area", "train", "tram"]

    classe_caminho = []

    arq = open(entrada)
    linhas = arq.readlines()
    for i in linhas:
        aux = i.split("	")
        aux2 = aux[1].replace("\n", "")
        aux2 = aux2.replace("/", "_")
        aux2 += ";"
        aux2 += aux[0]
        classe_caminho.append(aux2)

    classe_caminho.sort()

    spectrograms_directory = "./spec_evaluation_completo/"

    for c in classes:
        path_criacao = spectrograms_directory + "class_" + c
        if os.path.exists(path_criacao) == False:
            os.makedirs(path_criacao)

    for i in range(0, len(classe_caminho)):
        aux = classe_caminho[i].split(";")
        path = aux[1]
        classe_atual = aux[0]
        audio, sr = librosa.load(path)

        path_save = spectrograms_directory + "class_" + classe_atual + '/'

        espectrograma = librosa.stft(audio)

        plt.axis('off')
        librosa.display.specshow(librosa.amplitude_to_db(
            abs(espectrograma), ref=np.max), sr=22050, y_axis='log', x_axis='time')
        abs_path = path_save + f'{i}.png'
        print(abs_path)
        plt.savefig(abs_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close('all')
        gc.collect()


if __name__ == '__main__':
    main()
