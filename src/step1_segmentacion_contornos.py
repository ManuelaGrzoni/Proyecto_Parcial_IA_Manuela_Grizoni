import cv2
import numpy as np

# 🔹 CAMBIA ESTE NÚMERO POR EL ÍNDICE DE TU IVCAM (0, 1, 2...)
CAM_INDEX = 1


def recortar_bordes_negros(frame):
    h, w = frame.shape[:2]

    # Igual que en step2, 3, 4 y 5: 20% arriba y 20% abajo
    top = int(h * 0.20)
    bottom = int(h * 0.80)

    return frame[top:bottom, :]



def segmentar_tapete_verde(frame):
    """
    Recibe un frame en BGR y devuelve una máscara binaria
    donde el fondo verde está en negro y la carta en blanco.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de verde (un poco más amplio para cubrir variaciones del tapete)
    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)   # 255 = verde
    mask_not_green = cv2.bitwise_not(mask_green)              # 255 = NO verde (carta, etc.)

    # Limpieza morfológica (quita ruido)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_not_green, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    return mask_clean


def encontrar_contornos_cartas(mask, min_area=3000, max_area=200000):
    """
    A partir de la máscara (carta en blanco), encuentra contornos
    y se queda con los que tienen un área dentro de [min_area, max_area].
    Así ignoramos ruido pequeño y cosas demasiado grandes.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cartas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cartas.append(cnt)
    return cartas


def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error capturando frame.")
            break

        # 0) Quitamos bandas negras de arriba y abajo
        frame_recortado = recortar_bordes_negros(frame)

        # 1) Segmentamos tapete verde
        mask = segmentar_tapete_verde(frame_recortado)

        # 2) Buscamos contornos (cartas)
        contornos = encontrar_contornos_cartas(mask)

        # 3) Dibujamos contornos sobre una copia
        frame_contornos = frame_recortado.copy()
        cv2.drawContours(frame_contornos, contornos, -1, (0, 0, 255), 3)

        # 4) Mostramos ventanas
        cv2.imshow("Original recortado", frame_recortado)
        cv2.imshow("Mascara (cartas en blanco)", mask)
        cv2.imshow("Contornos detectados", frame_contornos)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
