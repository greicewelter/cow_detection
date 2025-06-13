import os
import shutil
import xml.etree.ElementTree as ET
import random
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd


base_dir = '/home/nome_usuario/cow_detection'  
xml_path = os.path.join(base_dir, 'annotations.xml')
imagens_dir = os.path.join(base_dir, 'imagens')
boxes_dir = os.path.join(base_dir, 'boxes')  
data_yaml_path = os.path.join(base_dir, 'data.yaml')

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
classes = ['cow']

modelo_nome = 'cow_detector'
modelo_path = os.path.join(base_dir, 'yolov8n.pt')


def converter_xml_para_yolo():
    if not os.path.exists(boxes_dir):
        os.makedirs(boxes_dir)
    else:
       
        if os.listdir(boxes_dir):
            print("Labels já convertidos encontrados, pulando conversão.")
            return

    print("Convertendo annotations.xml para formato YOLO...")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall('image'):
        name = image.get('name')
        width = int(image.get('width'))
        height = int(image.get('height'))

        image_path = os.path.join(imagens_dir, name)
        if not os.path.isfile(image_path):
            print(f"Aviso: imagem {name} não encontrada, pulando...")
            continue

        txt_name = os.path.splitext(name)[0] + '.txt'
        txt_path = os.path.join(boxes_dir, txt_name)

        with open(txt_path, 'w') as f:
            for box in image.findall('box'):
                label = box.get('label')
                if label not in classes:
                    continue
                class_id = classes.index(label)

                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))

                x_center = ((xtl + xbr) / 2) / width
                y_center = ((ytl + ybr) / 2) / height
                w = (xbr - xtl) / width
                h = (ybr - ytl) / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    print("Conversão concluída.")

def dividir_organizar_dados():
    # Criar pastas para treino, val, test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(imagens_dir, split), exist_ok=True)
        os.makedirs(os.path.join(imagens_dir, split, 'labels'), exist_ok=True)

    
    imagens_com_label = []
    for label_file in os.listdir(boxes_dir):
        if not label_file.endswith('.txt'):
            continue
        base_name = label_file[:-4] + '.png'  # assumindo extensão .png
        image_path = os.path.join(imagens_dir, base_name)
        if os.path.isfile(image_path):
            imagens_com_label.append(base_name)

    print(f"Total de imagens com labels: {len(imagens_com_label)}")

    random.shuffle(imagens_com_label)
    n = len(imagens_com_label)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_imgs = imagens_com_label[:n_train]
    val_imgs = imagens_com_label[n_train:n_train + n_val]
    test_imgs = imagens_com_label[n_train + n_val:]

    def mover_arquivos(lista, split):
        for img_name in lista:
            label_name = os.path.splitext(img_name)[0] + '.txt'

            src_img = os.path.join(imagens_dir, img_name)
            dst_img = os.path.join(imagens_dir, split, img_name)

            src_label = os.path.join(boxes_dir, label_name)
            dst_label = os.path.join(imagens_dir, split, 'labels', label_name)

            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_label, dst_label)

    mover_arquivos(train_imgs, 'train')
    mover_arquivos(val_imgs, 'val')
    mover_arquivos(test_imgs, 'test')

    print(f"Imagens e labels movidos para pastas train ({len(train_imgs)}), val ({len(val_imgs)}) e test ({len(test_imgs)}).")

def gerar_yaml():
    content = f"""train: {os.path.join(imagens_dir, 'train')}
val: {os.path.join(imagens_dir, 'val')}
test: {os.path.join(imagens_dir, 'test')}

nc: {len(classes)}
names: {classes}
"""
    with open(data_yaml_path, 'w') as f:
        f.write(content)
    print(f"Arquivo data.yaml gerado em: {data_yaml_path}")

def treinar_e_avalizar():
    print("Iniciando treinamento...")
    model = YOLO(modelo_path)
    model.train(
        data=data_yaml_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name=modelo_nome,
        val=True,
        exist_ok=True,
        patience=10,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0
    )

    print("Avaliando modelo...")
    val_metrics = model.val(data=data_yaml_path, split='val')

    if hasattr(val_metrics, 'results_dict'):
        mAP50 = val_metrics.results_dict.get('metrics/mAP50(B)', None)
        if mAP50 is not None and pd.notna(mAP50) and mAP50 > 0:
            acuracia = float(mAP50)
            erro = 1 - acuracia
            labels = ['Acurácia', 'Perda']
            sizes = [acuracia, erro ]
            colors = ['#66b3ff', '#ff6666']
            plt.figure(figsize=(6, 6))
            plt.pie(sizes,labels=labels, colors=colors, autopct='%1.1f%%', startangle=90) 
            plt.title("Desempenho do Modelo YOLO yolov8n.pt")
            plt.axis('equal')
            plt.tight_layout()

            grafico_path = os.path.join(base_dir, 'runs/detect', modelo_nome, 'grafico_acuracia_erro.png')
            plt.savefig(grafico_path)
            plt.show()
        else:
            print("Acurácia ausente ou zero. Gráfico não será gerado.")
    else:
        print("Métricas de validação não disponíveis.")

if __name__ == '__main__':
    converter_xml_para_yolo()
    dividir_organizar_dados()
    gerar_yaml()
    treinar_e_avalizar()
