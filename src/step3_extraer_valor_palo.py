import cv2
import numpy as np

# Ajusta este índice al de tu iVCam (0, 1, 2...)
CAM_INDEX = 1


# ------------------ PREPROCESADO ------------------ #

def recortar_bordes_negros(frame):
    h, w = frame.shape[:2]
    top = int(h * 0.20)
    bottom = int(h * 0.80)
    return frame[top:bottom, :]


def segmentar_tapete_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([30, 30, 30])
    upper_green = np.array([90, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_not_green = cv2.bitwise_not(mask_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_not_green, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    return mask_clean


def encontrar_contornos_cartas(mask, min_area=3000, max_area=200000):
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cartas = []
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            cartas.append(cnt)
    return cartas


def ordenar_esquinas(pts):
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")


def extraer_carta_normalizada(frame, contorno, ancho=200, alto=300):
    peri = cv2.arcLength(contorno, True)
    approx = cv2.approxPolyDP(contorno, 0.02 * peri, True)

    if len(approx) == 4:
        esquinas = ordenar_esquinas(approx)
    else:
        rect = cv2.minAreaRect(contorno)
        box = cv2.boxPoints(rect)
        esquinas = ordenar_esquinas(box)

    pts_dst = np.array([
        [0, 0],
        [ancho - 1, 0],
        [ancho - 1, alto - 1],
        [0, alto - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(esquinas, pts_dst)
    carta_warped = cv2.warpPerspective(frame, M, (ancho, alto))

    return carta_warped


# ------------------ NUEVO: BUSCAR AUTOMÁTICAMENTE LA ESQUINA BUENA ------------------ #

def extraer_valor_y_palo(carta_norm):
    """
    Busca la esquina (de las 4) donde hay más 'tinta' (símbolos)
    y luego separa valor y palo.
    Devuelve:
      - valor_roi (binaria)
      - palo_roi (binaria)
      - esquina_color (en color)
      - esquina_bin (binaria)
    """
    h, w = carta_norm.shape[:2]

    # Imagen binaria de toda la carta
    gray_full = cv2.cvtColor(carta_norm, cv2.COLOR_BGR2GRAY)
    _, thresh_full = cv2.threshold(
        gray_full, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Tamaño del recorte de esquina (AQUÍ HEMOS CAMBIADO LOS PORCENTAJES)
    # Antes: 0.30 * h y 0.25 * w  -> recorte pequeño, cortaba la A
    corner_h = int(0.40 * h)   # más alto
    corner_w = int(0.45 * w)   # más ancho

    # Coordenadas de las 4 esquinas posibles: (y, x)
    corners = [
        (0, 0),                             # arriba izquierda
        (0, w - corner_w),                  # arriba derecha
        (h - corner_h, 0),                  # abajo izquierda
        (h - corner_h, w - corner_w)        # abajo derecha
    ]

    best_score = -1
    best_yx = None

    # Buscamos la esquina con más píxeles blancos (más símbolo)
    for (y, x) in corners:
        roi = thresh_full[y:y + corner_h, x:x + corner_w]
        score = cv2.countNonZero(roi)  # nº de píxeles blancos
        if score > best_score:
            best_score = score
            best_yx = (y, x)

    # Usamos la mejor esquina encontrada
    y0, x0 = best_yx
    esquina_color = carta_norm[y0:y0 + corner_h, x0:x0 + corner_w]
    esquina_bin = thresh_full[y0:y0 + corner_h, x0:x0 + corner_w]

    # Separamos valor (arriba) y palo (abajo)
    h_esq, w_esq = esquina_bin.shape
    corte = int(h_esq * 0.55)

    valor_roi = esquina_bin[0:corte, :]
    palo_roi = esquina_bin[corte:h_esq, :]

    return valor_roi, palo_roi, esquina_color, esquina_bin


# ------------------ MAIN LOOP ------------------ #

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recortar_bordes_negros(frame)
        mask = segmentar_tapete_verde(frame)
        contornos = encontrar_contornos_cartas(mask)

        carta_norm = None

        if len(contornos) > 0:
            cnt = sorted(contornos, key=cv2.contourArea, reverse=True)[0]
            carta_norm = extraer_carta_normalizada(frame, cnt)

        if carta_norm is not None:
            valor_roi, palo_roi, esquina_raw, esquina_bin = extraer_valor_y_palo(carta_norm)

            cv2.imshow("Carta normalizada", carta_norm)
            cv2.imshow("Esquina ORIGINAL (color)", esquina_raw)
            cv2.imshow("Esquina BINARIA", esquina_bin)
            cv2.imshow("Valor (ROI)", valor_roi)
            cv2.imshow("Palo (ROI)", palo_roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
