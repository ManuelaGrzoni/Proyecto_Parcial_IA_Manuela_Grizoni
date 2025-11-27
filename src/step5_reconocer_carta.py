import cv2
import numpy as np
import os

CAM_INDEX = 1  # índice de tu iVCam

SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SRC_DIR)

PLANTILLAS_VALOR_DIR = os.path.join(ROOT_DIR, "plantillas", "valor")
PLANTILLAS_PALO_DIR  = os.path.join(ROOT_DIR, "plantillas", "palo")


def imread_unicode(path, flags):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img
    except Exception as e:
        print("Error leyendo:", path, "->", e)
        return None


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


def cargar_plantillas(directorio):
    plantillas = {}
    print("\nLeyendo plantillas desde:", directorio)
    if not os.path.isdir(directorio):
        print("⚠ Carpeta no encontrada.")
        return plantillas
    for fname in os.listdir(directorio):
        ruta = os.path.join(directorio, fname)
        img = imread_unicode(ruta, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("  ⚠ No se pudo leer:", ruta)
            continue
        clave = os.path.splitext(fname)[0]
        plantillas[clave] = img
        print("  ✔ Cargada plantilla:", clave, "->", img.shape)
    if not plantillas:
        print("⚠ No se cargó ninguna plantilla en", directorio)
    return plantillas


def reconocer_por_template(roi, plantillas):
    if not plantillas:
        return "desconocido", -1.0
    mejor_clave = "desconocido"
    mejor_score = -1
    for clave, templ in plantillas.items():
        roi_resized = cv2.resize(roi, (templ.shape[1], templ.shape[0]))
        res = cv2.matchTemplate(roi_resized, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > mejor_score:
            mejor_score = max_val
            mejor_clave = clave
    return mejor_clave, mejor_score


def main():
    print("SRC_DIR :", SRC_DIR)
    print("ROOT_DIR:", ROOT_DIR)
    print("VALOR DIR:", PLANTILLAS_VALOR_DIR)
    print("PALO DIR :", PLANTILLAS_PALO_DIR)

    plantillas_valor = cargar_plantillas(PLANTILLAS_VALOR_DIR)
    plantillas_palo  = cargar_plantillas(PLANTILLAS_PALO_DIR)

    print("\nPlantillas de valor cargadas:", list(plantillas_valor.keys()))
    print("Plantillas de palo cargadas :", list(plantillas_palo.keys()))

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rec = recortar_bordes_negros(frame)
        mask = segmentar_tapete_verde(frame_rec)
        contornos = encontrar_contornos_cartas(mask)

        salida = frame_rec.copy()

        for cnt in contornos:
            carta_norm = extraer_carta_normalizada(frame_rec, cnt)
            carta_orientada = orientar_carta(carta_norm)
            valor_roi, palo_roi = extraer_valor_y_palo(carta_orientada)

            cv2.imshow("Carta orientada", carta_orientada)
            cv2.imshow("Valor (ROI)", valor_roi)
            cv2.imshow("Palo (ROI)", palo_roi)

            valor, score_val = reconocer_por_template(valor_roi, plantillas_valor)
            palo, score_palo = reconocer_por_template(palo_roi, plantillas_palo)

            print(f"Scores -> valor: {score_val:.3f}   palo: {score_palo:.3f}")

            if score_val < 0.30:
                valor = "?"
            if score_palo < 0.30:
                palo = "?"

            nombre_carta = f"{valor} de {palo}"
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(salida, nombre_carta,
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Resultado", salida)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
