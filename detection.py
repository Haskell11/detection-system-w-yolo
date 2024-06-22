import cv2
import yaml
from ultralytics import YOLO
import numpy as np
import torch
#import airsim # способ ввода видео напрямую с камеры коптера из AirSim 
from collections import defaultdict



# Проверка наличия GPU и установка устройства
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Используемое устройство: {device}')

# Загрузка моделей YOLOv8 для детекции и сегментации
#detection_model = YOLO('yolov8m.pt') # Модель для детекции
#segmentation_model = YOLO('yolov8s-seg.pt')  # Модель для сегментации
#segmentation_model.to(device)
#detection_model.to(device)

# Загрузка моделей YOLO с обученными весами
model_detection = YOLO('E:\diplom\yolo8 detection\Runs\detect\Train4\weights\Best.pt')
#segmentation_model = YOLO('E:\diplom\yolo8 detection\Runs\detect\Train\weights\Best.pt')  # Модель для сегментации

model_detection.to(device)
#segmentation_model.to(device)



# Цвета для аннотаций
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]

# Функция для загрузки конфигурации датасета из YAML-файла
def load_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Загрузка конфигурации датасета
dataset_config = load_yaml_config('E:\diplom\yolo8 detection\my_dataset_yolo\data.yaml')

# История трекинга для траекторий
#track_history = defaultdict(lambda: [])

#  способ ввода видео напрямую с камеры коптера из AirSim 
# Подключение к клиенту AirSim
# client = airsim.MultirotorClient()
# client.confirmConnection()

# Функция для получения изображения с камеры (камера с индексом 3)
# def get_camera_image():
#     responses = client.simGetImages([airsim.ImageRequest(3, airsim.ImageType.Scene, False, False)])
#     response = responses[0]
#     img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
#     img_rgba = img1d.reshape(response.height, response.width, 4)
#     img_rgb = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2RGB)
#     return img_rgb

# Открытие видеофайла
video_path = 'E:\Bandicam\Videos\kopter_flight2.mp4'
capture = cv2.VideoCapture(video_path)

# Проверка, удалось ли открыть видеофайл
if not capture.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

# Получение параметров исходного видео
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture.get(cv2.CAP_PROP_FPS)
# Определение кодека и создание объекта VideoWriter для записи видео
output_path = 'E:\Bandicam\Videos\kopter_flight_detected3.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Пороговое значение расстояния (например, 10 метров)
#distance_threshold = 10

# Функция для оценки расстояния до объекта (упрощенно)
#def estimate_distance(box):
    # Предполагаем, что чем ближе объект, тем больше его размер в кадре
    # Используем площадь bounding box для оценки расстояния
   # x1, y1, x2, y2 = box
   # box_area = (x2 - x1) * (y2 - y1)
   # # Эмпирически подобранный коэффициент (нужно настроить в зависимости от камеры и высоты)
    #distance = 1000 / (box_area ** 0.5)
    #return distance



while True:
    ret, frame = capture.read()
    
    # Проверка, удалось ли прочитать кадр
    if not ret:
        break
    
    # Изменение размера кадра для ускорения обработки (например, до 640x360)
    frame_resized = cv2.resize(frame, (640, 360))

    # Обработка кадра с помощью модели YOLOv8 для детекции
    detection_results = model_detection(frame_resized, device=device)[0]

     # Обработка кадра с помощью модели YOLOv8 для сегментации
    #segmentation_results = segmentation_model(frame_resized, device=device)[0]

    for class_id, box, conf in zip(detection_results.boxes.cls.cpu().numpy(),
                                   detection_results.boxes.xyxy.cpu().numpy().astype(np.int32),
                                   detection_results.boxes.conf.cpu().numpy()):
        class_name = detection_results.names[int(class_id)]

        x1, y1, x2, y2 = box
        # Масштабирование координат рамок обратно до оригинального размера кадра
        x1 = int(x1 * frame_width / 640)
        y1 = int(y1 * frame_height / 360)
        x2 = int(x2 * frame_width / 640)
        y2 = int(y2 * frame_height / 360)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(class_id) % len(colors)], 2)
        confidence_text = f'{class_name} ({conf:.2%})'
        cv2.putText(frame,
                    confidence_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[int(class_id) % len(colors)], 2)
    
    # Обработка кадра с помощью модели YOLOv8 для сегментации
   # segmentation_results = segmentation_model(frame_resized, device=device)[0]

     # Обработка сегментации
   # if segmentation_results.masks is not None:
      #  masks = segmentation_results.masks.data.cpu().numpy()
       # classes = segmentation_results.boxes.cls.cpu().numpy()

       # for i, mask in enumerate(masks):
        #    class_id = int(classes[i])
        #    color = colors[class_id % len(colors)]
         #   resized_mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
          #  color_mask = np.zeros_like(frame)
          #  color_mask[resized_mask > 0] = color
          #  frame = cv2.addWeighted(frame, 1.0, color_mask, 0.5, 0)

        
    # Запись обработанного кадра в выходное видео
    out.write(frame)
    
    # Отображение кадра с детекцией
    cv2.imshow('YOLOv8 Detection', frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
capture.release()
out.release()
cv2.destroyAllWindows()
capture.release()
out.release()
cv2.destroyAllWindows()
