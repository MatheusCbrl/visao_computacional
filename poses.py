import cv2
from ultralytics import YOLO

# Função para capturar e processar o vídeo da webcam
def get_frame():
    model = YOLO("yolov8s.pt")  # Carregar o modelo de detecção de pose
    cap = cv2.VideoCapture(0)  # Captura de vídeo via webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar para o tamanho esperado pelo YOLO
        img = cv2.resize(frame, (640, 640))

        # Inferir usando o modelo YOLO de pose
        results = model(img)

        # Processar as detecções e obter as poses
        for result in results:
            keypoints = result.keypoints  # Coordenadas dos pontos da pose

            # Para cada pessoa detectada, desenhar os pontos e as conexões (esqueleto)
            for keypoint_set in keypoints.xy:  # Acessando diretamente as coordenadas x, y
                # Desenhar círculos nos pontos chave das articulações
                for i, keypoint in enumerate(keypoint_set):
                    # Extração das coordenadas x, y convertidas para inteiros
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Desenhar o ponto

                # Desenhar as conexões entre os pontos chave (esqueleto)
                skeleton_connections = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
                                        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]
                for connection in skeleton_connections:
                    part_a = connection[0]
                    part_b = connection[1]
                    x_a, y_a = int(keypoint_set[part_a][0]), int(keypoint_set[part_a][1])
                    x_b, y_b = int(keypoint_set[part_b][0]), int(keypoint_set[part_b][1])
                    cv2.line(frame, (x_a, y_a), (x_b, y_b), (0, 255, 255), 2)  # Desenhar linha entre os pontos

        # Codificar a imagem para ser exibida na página HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
