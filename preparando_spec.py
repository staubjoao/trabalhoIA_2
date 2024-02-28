import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import pickle
import gc


def carregar_audio_teste(path):
    audio, sr = librosa.load(path)

    janela_segundos = 5

    janela_amostras = int(janela_segundos * sr)

    janelas = librosa.util.frame(
        audio, frame_length=janela_amostras, hop_length=janela_amostras)

    return janelas.T


def carregar_audio(janela_segundos, path):
    audio, sr = librosa.load(path)

    janela_amostras = int(janela_segundos * sr)

    janelas = librosa.util.frame(
        audio, frame_length=janela_amostras, hop_length=janela_amostras)

    espectrogramas = []
    for janela in janelas.T:
        espectrograma = librosa.stft(janela)
        espectrogramas.append(espectrograma)
    return espectrograma


def spec_completo():
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

    spectrograms_directory = "./spec_evaluation_janela/"

    for c in classes:
        path_criacao = spectrograms_directory + "class_" + c
        if os.path.exists(path_criacao) == False:
            os.makedirs(path_criacao)

    print("spec_janela")

    for i, cc in enumerate(classe_caminho):
        aux = cc.split(";")
        path_criacao = spectrograms_directory + \
            "class_" + aux[0] + '/' + str(i)
        if os.path.exists(path_criacao) == False:
            os.makedirs(path_criacao)
        print(i, end=" ")

    for i in range(0, len(classe_caminho)):
        aux = classe_caminho[i].split(";")
        path = aux[1]
        classe_atual = aux[0]
        path = "./" + path
        img = carregar_audio_teste(path)

        path_save = spectrograms_directory + \
            "class_" + classe_atual + '/' + str(i) + '/'
        print(path_save)

        for j, janela in enumerate(img):
            espectrograma = librosa.stft(janela)

            plt.axis('off')
            librosa.display.specshow(librosa.amplitude_to_db(
                abs(espectrograma), ref=np.max), sr=22050, y_axis='log', x_axis='time')
            abs_path = path_save + f'{j}.png'
            plt.savefig(abs_path, bbox_inches='tight', pad_inches=0)
            plt.clf()
            plt.close('all')
            gc.collect()

        print(i, end=" ")

    print("spec_completo")
    spec_completo()


if __name__ == '__main__':
    main()
