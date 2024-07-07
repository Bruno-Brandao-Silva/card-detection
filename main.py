# Importações
import cv2
import albumentations as A
import os
from concurrent.futures import ThreadPoolExecutor
from simple_image_download import simple_image_download as simp
import shutil
import hashlib
from ultralytics import YOLO
import random
import keyboard
from typing import Literal

# Constantes de paths
download_dir = "simple_images"

img_positive_dir = "positive_images"
lb_positive_dir = "positive_labels"

img_negative_dir = "negative_images"
lb_negative_dir = "negative_labels"

img_augmented_dir = "augmented_images"
lb_augmented_dir = "augmented_labels"

# Palavras chaves do objeto
keywords = ["ace of spades", "ace of hearts", "ace of diamonds", "ace of clubs",
            "king of spades", "king of hearts", "king of diamonds", "king of clubs"]

# Palavras chaves negativas
negative_keywords = [
    "Nature", "Landscapes", "Animals", "Food", "Vehicles", "Buildings", "People",
    "Sports", "Technology", "Furniture", "Clothing", "Electronics", "Tools", "Toys",
    "Art", "Plants", "Cities", "Beaches", "Mountains", "Forests", "Oceans", "Rivers",
    "Birds", "Insects", "Roads", "Parks", "Bridges", "Monuments", "Statues", "Markets",
    "Festivals", "Shopping", "Libraries", "Museums", "Offices", "Schools", "Universities",
    "Airports", "Trains", "Boats", "Bicycles", "Skyscrapers", "Streets", "Concerts",
    "Theaters", "Farms", "Gardens", "Deserts", "Snow", "cards two", "cards three",
    "cards four", "cards five", "cards six", "cards seven", "cards eight", "cards nine",
    "cards ten", "cards jack", "cards queen", "cards joker", "cards back"
]

# Definindo a transformação
transform = A.Compose([
    A.SafeRotate(limit=40, p=0.5, border_mode=cv2.BORDER_CONSTANT,
                 value=[0, 0, 0]),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.4),
    A.Blur(p=0.4),
    A.RGBShift(p=0.4),
    A.HueSaturationValue(p=0.4),
    A.RandomFog(p=0.4),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Função para mover arquivos para a pasta de download
def move_files_to_download_root_dir(source_dir):
    for root, dirs, files in os.walk(source_dir, topdown=False):
        if root == source_dir:
            continue
        for file_name in files:
            old_path = os.path.join(root, file_name)
            new_path = os.path.join(source_dir, file_name)
            if os.path.exists(new_path):
                base, extension = os.path.splitext(file_name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(
                        source_dir, f"{base}_{counter}{extension}")
                    counter += 1
            shutil.move(old_path, new_path)
        os.rmdir(root)


# Função para calcular o hash de um arquivo
def calculate_file_hash(source_dir):
    hasher = hashlib.sha256()
    with open(source_dir, 'rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(4096), b""):
            hasher.update(bloco)
    return hasher.hexdigest()


# Função para remover duplicatas usando hash
def remove_duplicates(source_dir):
    hashes = {}
    removed_files = 0
    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hash = calculate_file_hash(file_path)
            if file_hash in hashes:
                os.remove(file_path)
                removed_files += 1
            else:
                hashes[file_hash] = file_path
    return removed_files


# Função para baixar imagens do objeto
def donwload_postive_images():
    response = simp.simple_image_download()
    for kw in keywords:
        response.download(keywords=kw, limit=50)
    move_files_to_download_root_dir(download_dir)
    print(f"Total de arquivos removidos: {remove_duplicates(download_dir)}")
    os.rename(download_dir, img_positive_dir)


# Função para baixar imagens negativas e criar labels vazias
def download_negative_images():
    response = simp.simple_image_download()
    for kw in negative_keywords:
        response.download(keywords=kw, limit=15)
    move_files_to_download_root_dir(download_dir)
    print(f"Total de arquivos removidos: {remove_duplicates(download_dir)}")
    os.rename(download_dir, img_negative_dir)
    os.makedirs(lb_negative_dir, exist_ok=True)
    for file in os.listdir(img_negative_dir):
        base, extension = os.path.splitext(file)
        with open(os.path.join(lb_negative_dir, f"{base}.txt"), 'w') as file:
            pass
    # Checa se a imagem está gerando Warnings
    for img_name in os.listdir(img_negative_dir):
        if img_name.endswith(".png"):
            img_path = os.path.join(img_negative_dir, img_name)
            # print(img_name) check for libpng warning:
            img = cv2.imread(img_path)


# Função para carregar anotações
def load_annotations(label_path):
    with open(label_path, 'r') as file:
        bboxes = []
        class_labels = []
        for line in file:
            class_id, center_x, center_y, width, height = map(
                float, line.strip().split())
            bboxes.append([center_x, center_y, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels


# Função para salvar anotações
def save_annotations(label_path, bboxes, class_labels):
    with open(label_path, 'w') as file:
        for bbox, class_label in zip(bboxes, class_labels):
            file.write(f"{class_label} {' '.join(map(str, bbox))}\n")


# Função para processar e aumentar uma única imagem
def process_image(img_name, _input_dir, _label_dir, num_augmentations):
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(_input_dir, img_name)
        label_path = os.path.join(
            _label_dir, os.path.splitext(img_name)[0] + '.txt')

        # Carregar imagem e anotações
        image = cv2.imread(image_path)
        bboxes, class_labels = load_annotations(label_path)

        for i in range(num_augmentations):
            # Aplicar transformação
            transformed = transform(
                image=image, bboxes=bboxes, class_labels=class_labels)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['class_labels']

            # Salvar imagem aumentada
            augmented_image_name = f"aug_{i}_{_input_dir}_{img_name}"
            augmented_image_path = os.path.join(
                img_augmented_dir, augmented_image_name)
            cv2.imwrite(augmented_image_path, transformed_image)

            # Salvar anotações aumentadas
            augmented_label_name = f"aug_{i}_{_input_dir}_{os.path.splitext(img_name)[0]}.txt"
            augmented_label_path = os.path.join(
                lb_augmented_dir, augmented_label_name)
            save_annotations(augmented_label_path,
                             transformed_bboxes, transformed_class_labels)


# Função para aumentar imagens de um diretório em paralelo
def augment_images(_input_dir, _label_dir, num_augmentations, delete_old=False):
    if delete_old:
        if os.path.exists(img_augmented_dir):
            shutil.rmtree(img_augmented_dir)
            os.makedirs(img_augmented_dir)
        else:
            os.makedirs(img_augmented_dir)
        if os.path.exists(lb_augmented_dir):
            shutil.rmtree(lb_augmented_dir)
            os.makedirs(lb_augmented_dir)
        else:
            os.makedirs(lb_augmented_dir)

    with ThreadPoolExecutor() as executor:
        img_names = [img_name for img_name in os.listdir(
            _input_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]
        executor.map(process_image, img_names, [_input_dir]*len(img_names), [
                     _label_dir]*len(img_names), [num_augmentations]*len(img_names))


# Função para criar tuplas de nome de arquivo e extensão
def list_files_with_separated_extensions(directory):
    files = []
    for file in os.listdir(directory):
        filename, extension = os.path.splitext(file)
        files.append((filename, extension))
    return files


# Função para copiar todos os arquivos de um diretório para outro
def copy_files_to(from_dir, to_dir):
    for file in os.listdir(from_dir):
        shutil.copy(os.path.join(from_dir, file), os.path.join(to_dir, file))


# Função que copia e separa os arquivos de treino e validação
def create_train_val_data():
    if os.path.exists("train"):
        shutil.rmtree("train")
    if os.path.exists("val"):
        shutil.rmtree("val")

    os.makedirs("train/images", exist_ok=True)
    os.makedirs("train/labels", exist_ok=True)
    os.makedirs("val/images", exist_ok=True)
    os.makedirs("val/labels", exist_ok=True)

    copy_files_to("positive_images", "train/images")
    copy_files_to("positive_labels", "train/labels")

    copy_files_to("negative_images", "train/images")
    copy_files_to("negative_labels", "train/labels")

    copy_files_to("augmented_images", "train/images")
    copy_files_to("augmented_labels", "train/labels")

    tuple_files = list_files_with_separated_extensions("train/images")
    num_files = len(tuple_files)
    files_to_val = random.sample(tuple_files, int(num_files * 0.2))

    for file in files_to_val:
        shutil.move(os.path.join("train/images",
                    f"{file[0]}{file[1]}"), os.path.join("val/images", f"{file[0]}{file[1]}"))
        shutil.move(os.path.join("train/labels",
                    f"{file[0]}.txt"), os.path.join("val/labels", f"{file[0]}.txt"))

    print(
        f"Train images: {num_files-len(files_to_val)} | Val images: {len(files_to_val)} | Total images: {num_files}")


# Função de treinamento e avaliação do modelo YOLOv8
def train_yolov8(_model: Literal['yolov8n.pt', 'yolov8s.pt'],
                 img_size, batch_size, epochs):
    # model = YOLO("C:/Users/bruno/Desktop/Bruno/DEV/card-detection/card-detection/runs/detect/train3/weights/last.pt")
    # model.train(resume=True)

    model = YOLO(_model)
    model.train(data='data_custom.yaml',
                imgsz=img_size,
                batch=batch_size,
                epochs=epochs,
                device='cuda',
                )
    return


# Função que gera o path absoluto do modelo
def get_model(version):
    if version == 0:
        return
    if version == 1:
        return os.path.join(os.getcwd(), "runs", "detect", "train", "weights", "best.pt")
    else:
        return os.path.join(os.getcwd(), "runs", "detect", f"train{version}", "weights", "best.pt")


# Função para processar e armazenar dados dos jogadores
def process_data(r, names):
    object_counts = {}

    for box in r.boxes:
        class_id = int(box.cls[0].item())
        label = names[class_id]
        if label in object_counts:
            object_counts[label] += 1
        else:
            object_counts[label] = 1

    # Armazena o resultado no dicionário do jogador
    return object_counts


# Função para comparar e mostrar o vencedor
def compare_players(player1_data, player2_data):
    if not player1_data or not player2_data:
        print("Dados insuficientes para comparação")
        return

    player1_score = player1_data['ace'] if 'ace' in player1_data else 0
    player2_score = player2_data['ace'] if 'ace' in player2_data else 0

    if player1_score > player2_score:
        print("Player 1 vence!")
    elif player2_score > player1_score:
        print("Player 2 vence!")
    else:
        print("Empate!")


def test_model(model_version=1, img_size=1920, confiance=0.8):
    model = YOLO(get_model(model_version))

    player1_data = {}
    player2_data = {}

    results = model.predict(source="0", show=True,
                            stream=True, conf=confiance, verbose=False,
                            imgsz=img_size)
    for r in results:

        if keyboard.is_pressed('1'):
            print("Registrando dados para o Player 1")
            player1_data = process_data(r, model.names)
            print("Player 1: ", player1_data)

        elif keyboard.is_pressed('2'):
            print("Registrando dados para o Player 2")
            player2_data = process_data(r, model.names)
            print("Player 2: ", player2_data)

        elif keyboard.is_pressed('v'):
            print("Player 1: ", player1_data)
            print("Player 2: ", player2_data)

        elif keyboard.is_pressed('r'):
            player1_data = {}
            player2_data = {}
            print("Dados resetados")

        elif keyboard.is_pressed('c'):
            compare_players(player1_data, player2_data)
        elif keyboard.is_pressed('q'):
            return


def main():
    # # # Baixar imagens positivas
    # donwload_postive_images()

    # # # Baixar imagens negativas
    # download_negative_images()

    # # # Aumentar de dados nas imagens positivas
    # augment_images(img_positive_dir, lb_positive_dir, 10, delete_old=True)

    # # # Aumentar de dados nas imagens negativas
    # augment_images(img_negative_dir, lb_negative_dir, 1)

    # # Separar os dados de treino e validação
    # create_train_val_data()

    # # # Treinar o modelo
    # train_yolov8("yolov8n.pt", 512, 20, 200)
    # train_yolov8("yolov8s.pt", 512, 20, 200)


    # # # Comandos 
    # # Para sair do loop de detecção, pressione e segure 'q'
    # # Para registrar dados para o Player 1, pressione '1'
    # # Para registrar dados para o Player 2, pressione '2'
    # # Para visualizar os dados registrados, pressione 'v'
    # # Para resetar os dados registrados, pressione 'r'
    # # Para comparar os dados registrados, pressione 'c'



    # # # Teste de detecção e jogo
    # # model_version=1   -> Modelo yolo8n (mais rápido) 
    # # model_version=2   -> Modelo yolo8s (mais preciso e lento)
    # # img_size=1920     -> Tamanho da imagem para detecção (quanto maior, mais preciso e lento)
    # # confiance=0.8     -> Confiança da detecção (quanto maior, mais preciso e menos detecções)
    test_model(model_version=1, img_size=1920, confiance=0.8)


if __name__ == '__main__':
    main()
