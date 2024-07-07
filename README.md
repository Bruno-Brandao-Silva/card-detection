# Projeto de Detecção de Cartas com YOLOv8

Este projeto realiza a detecção de cartas de baralho usando a rede neural YOLOv8. Ele inclui scripts para download de imagens, augmentação de dados, treinamento e avaliação do modelo, bem como um mini-jogo para comparar resultados entre dois jogadores.

## Funcionalidades

- Download de imagens positivas (cartas de baralho) e imagens negativas.
- Augmentação de dados para aumentar a quantidade de imagens de treinamento.
- Treinamento do modelo YOLOv8 com as imagens processadas.
- Avaliação do modelo com um sistema de comparação de jogadores.

## Requisitos

- Python 3.x
- OpenCV
- Albumentations
- Simple-Image-Download
- Ultralytics YOLO
- Concurrent Futures
- Keyboard

## Estrutura do Projeto

```
.
├── positive_images/         # Imagens positivas
├── positive_labels/         # Anotações das imagens positivas
├── negative_images/         # Imagens negativas
├── negative_labels/         # Anotações das imagens negativas (vazias)
├── augmented_images/        # Imagens aumentadas
├── augmented_labels/        # Anotações das imagens aumentadas
├── train/                   # Dados de treino
│   ├── images/
│   └── labels/
├── val/                     # Dados de validação
│   ├── images/
│   └── labels/
├── main.py                  # Script principal
├── README.md                # Este arquivo
└── data_custom.yaml         # Arquivo de configuração de dados para YOLOv8
```

## Como Usar

1. **Baixar Imagens Positivas**:
   ```python
   donwload_postive_images()
   ```

2. **Baixar Imagens Negativas**:
   ```python
   download_negative_images()
   ```

3. **Augmentação de Dados**:
   ```python
   augment_images(img_positive_dir, lb_positive_dir, 10, delete_old=True)
   augment_images(img_negative_dir, lb_negative_dir, 1)
   ```

4. **Criar Dados de Treino e Validação**:
   ```python
   create_train_val_data()
   ```

5. **Treinar o Modelo YOLOv8**:
   ```python
   train_yolov8("yolov8n.pt", 512, 20, 200)
   ```

6. **Testar o Modelo e Jogar**:
   ```python
   test_model(model_version=1, img_size=1920, confiance=0.8)
   ```

## Comandos de Jogo

- `1`: Registrar dados para o Player 1
- `2`: Registrar dados para o Player 2
- `v`: Visualizar os dados registrados
- `r`: Resetar os dados registrados
- `c`: Comparar os dados registrados
- `q`: Sair do loop de detecção
