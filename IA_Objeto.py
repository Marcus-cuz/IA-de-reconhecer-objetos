import tensorflow as tf
import numpy as np
import cv2
import time
from object_detection.utils import label_map_util

# Carregar o modelo
model_dir = 'path/to/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
detect_fn = tf.saved_model.load(model_dir)

# Carregar o rótulo dos objetos
labels_path = 'path/to/mscoco_label_map.pbtxt'
category_index = {}

with open(labels_path, 'r') as f:
    label_map = label_map_util.load_labelmap(f)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

# Função para desenhar caixas delimitadoras
def draw_boxes(image, boxes, classes, scores, category_index, threshold=0.5):
    for i in range(boxes.shape[0]):
        if scores[i] > threshold:
            box = boxes[i]
            class_name = category_index[classes[i]]['name']
            y_min, x_min, y_max, x_max = box
            (left, right, top, bottom) = (x_min * image.shape[1], x_max * image.shape[1],
                                          y_min * image.shape[0], y_max * image.shape[0])
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
            label = f'{class_name}: {int(scores[i] * 100)}%'
            cv2.putText(image, label, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Capturar imagem da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocessar a imagem
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.float32)
    
    # Realizar a detecção
    detections = detect_fn(input_tensor)
    
    # Extrair informações da detecção
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # Detecção de classes
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # Desenhar caixas delimitadoras na imagem
    draw_boxes(frame, detections['detection_boxes'], detections['detection_classes'], detections['detection_scores'], category_index)

    # Mostrar imagem com detecções
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
