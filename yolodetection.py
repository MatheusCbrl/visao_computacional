import cv2
from ultralytics import YOLO

# Função para capturar e processar o vídeo da webcam
def get_frame():
    model = YOLO("yolov8s.pt")  # Carregar o modelo com os pesos corretos - nano(n), small(s), medium(m), large(l), and extra large(x).
    cap = cv2.VideoCapture(0)  # Captura de vídeo via webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar para o tamanho esperado pelo YOLO
        img = cv2.resize(frame, (640, 640))

        # Inferir usando o modelo YOLO
        results = model(img)

        # Processar as detecções e obter caixas e segmentações
        for result in results:
            boxes = result.boxes  # Coordenadas das caixas delimitadoras

            # Desenhar as caixas delimitadoras para todos os objetos detectados
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Extração dos limites
                confidence = box.conf.item()  # Confiança da detecção
                class_name = model.names[int(box.cls)]  # Nome da classe detectada

                # Desenhar caixa delimitadora para qualquer objeto detectado
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Codificar a imagem para ser exibida na página HTML
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
