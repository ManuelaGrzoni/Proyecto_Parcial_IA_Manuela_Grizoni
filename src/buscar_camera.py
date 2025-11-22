import cv2

print("Buscando cámaras disponibles...\n")

for i in range(5):  # prueba cámaras 0,1,2,3,4
    print(f"Probando cámara {i}...")
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  ❌ Cámara {i} no disponible.\n")
        continue

    ret, frame = cap.read()
    if not ret:
        print(f"  ❌ Cámara {i} sin imagen.\n")
        cap.release()
        continue

    print(f"  ✅ Cámara {i} está funcionando. Mírala en ventana.\n")
    cv2.imshow(f"CAMARA {i}", frame)
    key = cv2.waitKey(0)

    # cerrar la ventana para seguir probando otras
    cv2.destroyAllWindows()
    cap.release()
