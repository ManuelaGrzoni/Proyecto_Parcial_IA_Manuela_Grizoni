import cv2

# PRUEBA DIRECTA CON LA CÁMARA 1 (suele ser iVCam)
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("No se pudo abrir la cámara 1")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error capturando frame")
        break

    cv2.imshow("Camara seleccionada", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
