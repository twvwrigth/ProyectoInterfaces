from pathlib import Path
import sys
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

RUTA_CSV = ROOT_DIR / "data" / "dataset_lsm.csv"
RUTA_MODELO = ROOT_DIR / "data" / "modelo_lsm.pkl"


def main():
    if not RUTA_CSV.exists():
        print(f"[ERROR] No existe {RUTA_CSV}. Ejecuta primero 'python scripts/capturar_dataset.py'.")
        return

    df = pd.read_csv(RUTA_CSV)
    if len(df) < 50:
        print("[ERROR] Muy pocas muestras en el dataset. Captura al menos 100 muestras por letra.")
        return

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    clases_unicas = sorted(df["label"].unique())
    print(f"Total de muestras: {len(df)} | Letras registradas ({len(clases_unicas)}): {clases_unicas}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)

    print(f"\nPrecisión en conjunto de prueba: {precision * 100:.2f}%\n")
    print("Reporte detallado por letra:")
    print(classification_report(y_test, y_pred))

    paquete = {
        "model": modelo,
        "classes": modelo.classes_,
    }
    joblib.dump(paquete, RUTA_MODELO)
    print(f"[OK] Modelo guardado exitosamente en: {RUTA_MODELO}")


if __name__ == "__main__":
    main()