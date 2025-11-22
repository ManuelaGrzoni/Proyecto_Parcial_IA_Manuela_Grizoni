import cv2
import numpy as np

# 🔹 CAMBIA ESTE NÚMERO POR EL ÍNDICE DE TU IVCAM (0, 1, 2...)
CAM_INDEX = 1


# ------------------ PREPROCESADO ------------------ #

def recortar_bordes_negros(frame):
    """
    Recorta por arriba y por abajo para quitar las bandas negras
    que pone iVCam. Ajusta los porcentajes si hace falta.
    """
    h, w = frame.shape[:2]
    top = int(h * 0.20)      # 20% arriba
    bottom = int(h * 0.80)   # 20% abajo
    return frame[top:bottom, :]


def segmentar_tapete_verde(frame):
    """
    Devuelve una máscara binaria donde el tapete verde es negro
    y la carta es blanca.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)   # 255 = verde
    mask_not_green = cv2.bitwise_not(mask_green)              # 255 = NO verde

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_not_green, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    return mask_clean


def encontrar_contornos_cartas(mask, min_area=3000, max_area=200000):
    """
    Devuelve la lista de contornos cuya área está entre min_area y max_area.
    """
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cartas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cartas.append(cnt)
    return cartas


# ------------------ PERSPECTIVA / WARP ------------------ #

def ordenar_esquinas(pts):
    """
    Ordena 4 puntos (esquinas) en el orden:
    top-left, top-right, bottom-right, bottom-left
    """
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def extraer_carta_normalizada(frame, contorno, ancho=200, alto=300):
    """
    A partir del contorno de la carta, calcula las esquinas y aplica
    una transformación de perspectiva para obtener la carta 'recta'
    con tamaño fijo (ancho x alto).
    """

    # Aproximamos el contorno a un polígono
    peri = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.02 * peri, True)

    if len(approx) == 4:
        # Tenemos 4 esquinas "limpias"
        esquinas = ordenar_esquinas(approx)
    else:
        # Si no hay 4 puntos (por ruido/oclusiones), usamos un rectángulo mínimo
        rect = cv2.minAreaRect(contorno)          # centro, (w,h), ángulo
        box = cv2.boxPoints(rect)                 # 4 esquinas
        esquinas = ordenar_esquinas(box)

    # Definimos las coordenadas destino (carta recta)
    pts_dst = np.array([
        [0, 0],
        [ancho - 1, 0],
        [ancho - 1, alto - 1],
        [0, alto - 1]
    ], dtype="float32")

    # Matriz de transformación y warp
    M = cv2.getPerspectiveTransform(esquinas, pts_dst)
    carta_warped = cv2.warpPerspective(frame, M, (ancho, alto))

    return carta_warped


# ------------------ MAIN LOOP ------------------ #

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

        # 0) Recortar bandas negras
        frame_recortado = recortar_bordes_negros(frame)

        # 1) Máscara
        mask = segmentar_tapete_verde(frame_recortado)

        # 2) Contornos de cartas
        contornos = encontrar_contornos_cartas(mask)

        # 3) Dibujar contornos para depurar
        frame_contornos = frame_recortado.copy()
        cv2.drawContours(frame_contornos, contornos, -1, (0, 0, 255), 3)

        # 4) Si hay al menos una carta, extraemos la primera
        carta_norm = None
        if len(contornos) > 0:
            # Elegimos el contorno más grande (la carta principal)
            contornos_ordenados = sorted(contornos, key=cv2.contourArea, reverse=True)
            cnt_carta = contornos_ordenados[0]

            carta_norm = extraer_carta_normalizada(frame_recortado, cnt_carta,
                                                   ancho=200, alto=300)

        # --- Mostrar ventanas ---
        cv2.imshow("Original recortado", frame_recortado)
        cv2.imshow("Mascara (cartas en blanco)", mask)
        cv2.imshow("Contornos detectados", frame_contornos)

        if carta_norm is not None:
            cv2.imshow("Carta normalizada", carta_norm)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
