import cv2
import numpy as np

CAM_INDEX = 1  # tu iVCam


# --------- PREPROCESADO BÁSICO --------- #

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
    return [c for c in contornos if min_area < cv2.contourArea(c) < max_area]


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
    dst = np.array([[0, 0], [ancho - 1, 0],
                    [ancho - 1, alto - 1], [0, alto - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(esquinas, dst)
    return cv2.warpPerspective(frame, M, (ancho, alto))


# --------- ORIENTAR CARTA (MISMO QUE STEP5) --------- #

def orientar_carta(carta_norm):
    h, w = carta_norm.shape[:2]
    gray = cv2.cvtColor(carta_norm, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    corner_h = int(0.40 * h)
    corner_w = int(0.45 * w)

    corners = [
        thresh[0:corner_h, 0:corner_w],                # arriba izq
        thresh[0:corner_h, w - corner_w:w],            # arriba der
        thresh[h - corner_h:h, 0:corner_w],            # abajo izq
        thresh[h - corner_h:h, w - corner_w:w]         # abajo der
    ]

    scores = [cv2.countNonZero(c) for c in corners]
    idx = int(np.argmax(scores))

    if idx == 0:
        return carta_norm
    elif idx == 1:
        return cv2.rotate(carta_norm, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif idx == 2:
        return cv2.rotate(carta_norm, cv2.ROTATE_180)
    else:
        return cv2.rotate(carta_norm, cv2.ROTATE_90_CLOCKWISE)


# --------- VALOR Y PALO --------- #

def extraer_valor_y_palo_debug(carta_orientada):
    """versión para debug: devuelve también la esquina."""
    h, w = carta_orientada.shape[:2]

    corner = carta_orientada[0:int(0.30 * h), 0:int(0.35 * w)]

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ch, cw = binaria.shape
    corte = int(ch * 0.55)

    valor = binaria[0:corte, :]
    palo = binaria[corte:ch, :]

    return valor, palo, corner, binaria


# --------- MAIN --------- #

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = recortar_bordes_negros(frame)
        mask = segmentar_tapete_verde(frame)
        contornos = encontrar_contornos_cartas(mask)

        if contornos:
            cnt = sorted(contornos, key=cv2.contourArea, reverse=True)[0]
            carta_norm = extraer_carta_normalizada(frame, cnt)
            carta_orientada = orientar_carta(carta_norm)
            valor, palo, esquina_color, esquina_bin = extraer_valor_y_palo_debug(carta_orientada)

            cv2.imshow("Carta normalizada", carta_norm)
            cv2.imshow("Carta orientada", carta_orientada)
            cv2.imshow("Esquina color", esquina_color)
            cv2.imshow("Esquina binaria", esquina_bin)
            cv2.imshow("Valor (ROI)", valor)
            cv2.imshow("Palo (ROI)", palo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
