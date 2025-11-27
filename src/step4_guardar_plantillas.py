import cv2
import numpy as np
import os

CAM_INDEX = 1  # índice de tu iVCam

# ==== AJUSTA ESTOS NOMBRES ANTES DE EJECUTAR ====
NOMBRE_VALOR = "9"           # "A", "2", ..., "K"
NOMBRE_PALO  = "trebol"   # "picas", "corazones", "trebol", "diamantes"
# =================================================


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
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ], dtype="float32")


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


def orientar_carta(carta_norm):
    h, w = carta_norm.shape[:2]
    gray = cv2.cvtColor(carta_norm, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    corner_h = int(0.40 * h)
    corner_w = int(0.45 * w)
    corners = [
        thresh[0:corner_h, 0:corner_w],
        thresh[0:corner_h, w - corner_w:w],
        thresh[h - corner_h:h, 0:corner_w],
        thresh[h - corner_h:h, w - corner_w:w]
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


def extraer_valor_y_palo(carta_orientada):
    h, w = carta_orientada.shape[:2]
    corner = carta_orientada[0:int(0.40 * h), 0:int(0.45 * w)]
    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ch, cw = binaria.shape
    corte = int(ch * 0.55)
    valor = binaria[0:corte, :]
    palo = binaria[corte:ch, :]
    return valor, palo


def main():
    print("Directorio de trabajo actual:", os.getcwd())
    os.makedirs("plantillas/valor", exist_ok=True)
    os.makedirs("plantillas/palo", exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = recortar_bordes_negros(frame)
        mask = segmentar_tapete_verde(frame)
        contornos = encontrar_contornos_cartas(mask)

        valor_t = palo_t = None

        if contornos:
            carta_norm = extraer_carta_normalizada(frame, contornos[0])
            carta_orientada = orientar_carta(carta_norm)
            valor, palo = extraer_valor_y_palo(carta_orientada)

            valor_t = cv2.resize(valor, (60, 80))
            palo_t  = cv2.resize(palo,  (60, 60))

            cv2.imshow("Carta orientada", carta_orientada)
            cv2.imshow("Valor plantilla", valor_t)
            cv2.imshow("Palo plantilla", palo_t)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s') and valor_t is not None and palo_t is not None:
            ruta_valor = f"plantillas/valor/{NOMBRE_VALOR}.png"
            ruta_palo  = f"plantillas/palo/{NOMBRE_PALO}.png"
            cv2.imwrite(ruta_valor, valor_t)
            cv2.imwrite(ruta_palo, palo_t)
            print("Plantillas guardadas en:")
            print("  ", ruta_valor)
            print("  ", ruta_palo)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
