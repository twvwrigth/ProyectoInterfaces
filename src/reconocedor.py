from collections import Counter, deque
from pathlib import Path
import joblib
import numpy as np


class ReconocedorLSM:
    """Clasifica el vector de landmarks en una letra de LSM con suavizado temporal."""

    def __init__(self, ruta_modelo: str = "data/modelo_lsm.pkl", ventana_suavizado: int = 8):
        self.ruta_modelo = Path(ruta_modelo)
        self.modelo = None
        self.clases = []
        self.historial = deque(maxlen=ventana_suavizado)
        self.modelo_cargado = False
        self.cargar_modelo()

    def cargar_modelo(self) -> bool:
        if not self.ruta_modelo.exists():
            self.modelo_cargado = False
            return False

        paquete = joblib.load(self.ruta_modelo)
        self.modelo = paquete["model"]
        self.clases = list(paquete["classes"])
        self.modelo_cargado = True
        return True

    def predecir(self, vector_63: np.ndarray):
        """
        Recibe un vector (63,) y retorna (letra_suavizada, confianza_promedio).
        Si no hay modelo entrenado o vector válido, retorna (None, 0.0).
        """
        if not self.modelo_cargado or vector_63 is None:
            return None, 0.0

        entrada = vector_63.reshape(1, -1)
        probabilidades = self.modelo.predict_proba(entrada)[0]
        idx_max = int(np.argmax(probabilidades))
        letra_actual = str(self.clases[idx_max])
        confianza_actual = float(probabilidades[idx_max])

        self.historial.append((letra_actual, confianza_actual))

        # Votación mayoritaria en los últimos frames
        conteo = Counter([item[0] for item in self.historial])
        letra_ganadora, _ = conteo.most_common(1)[0]

        confianzas_ganadora = [conf for let, conf in self.historial if let == letra_ganadora]
        confianza_promedio = float(np.mean(confianzas_ganadora))

        return letra_ganadora, confianza_promedio

    def limpiar_historial(self):
        self.historial.clear()