import csv
from pathlib import Path
import sys
import cv2

# Permitir importar módulos desde src/ al ejecutar desde la raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.landmarks import ExtractorLandmarks
from src.tutor import TutorLSM

RUTA_CSV = ROOT_DIR / "data" / "dataset_lsm.csv"
MUESTRAS_POR_RAFAGA = 100


def inicializar_csv():
    RUTA_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not RUTA_CSV.exists():
        encabezados = ["label"] + [f"f{i}" for i in range(63)]
        with open(RUTA_CSV, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(encabezados)


def contar_muestras_por_letra() -> dict:
    conteos = {letra: 0 for letra in TutorLSM.LETRAS_LSM}
    if not RUTA_CSV.exists():
        return conteos
    with open(RUTA_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Saltar encabezado
        for fila in reader:
            if fila and fila[0] in conteos:
                conteos[fila[0]] += 1
    return conteos


def main():
    inicializar_csv()
    conteos = contar_muestras_por_letra()
    letras = TutorLSM.LETRAS_LSM
    idx_actual = 0

    extractor = ExtractorLandmarks()
    cap = cv2.VideoCapture(0)

    capturando = False
    muestras_restantes = 0

    print("=== RECOLECTOR DE DATASET LSM ===")
    print("Controles en ventana:")
    print("  [ESPACIO] -> Capturar 100 muestras de la letra actual")
    print("  [N]       -> Siguiente letra")
    print("  [P]       -> Letra anterior")
    print("  [Q]       -> Salir")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_anotado, vector, mano_detectada = extractor.procesar_frame(frame, dibujar=True)
        letra_actual = letras[idx_actual]

        if capturando and mano_detectada and vector is not None:
            with open(RUTA_CSV, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([letra_actual] + vector.tolist())
            conteos[letra_actual] += 1
            muestras_restantes -= 1
            if muestras_restantes <= 0:
                capturando = False
                print(f"[OK] Letra '{letra_actual}' completada. Total acumulado: {conteos[letra_actual]}")

        # HUD informativo
        cv2.rectangle(frame_anotado, (10, 10), (630, 115), (20, 24, 38), -1)
        cv2.putText(
            frame_anotado,
            f"Letra objetivo: {letra_actual} ({idx_actual + 1}/{len(letras)}) | Muestras: {conteos[letra_actual]}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (74, 222, 128),
            2,
        )

        estado_txt = (
            f"CAPTURANDO... Faltan {muestras_restantes}"
            if capturando
            else "[ESPACIO] Grabar 100 | [N] Sig | [P] Ant | [Q] Salir"
        )
        color_estado = (0, 165, 255) if capturando else (226, 232, 240)
        cv2.putText(frame_anotado, estado_txt, (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color_estado, 2)

        cv2.imshow("Captura de Dataset - LSM", frame_anotado)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q") or tecla == ord("Q"):
            break
        elif tecla == ord(" ") and not capturando:
            capturando = True
            muestras_restantes = MUESTRAS_POR_RAFAGA
        elif tecla in (ord("n"), ord("N")) and not capturando:
            idx_actual = (idx_actual + 1) % len(letras)
        elif tecla in (ord("p"), ord("P")) and not capturando:
            idx_actual = (idx_actual - 1) % len(letras)

    cap.release()
    extractor.cerrar()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()