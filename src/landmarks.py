import cv2
import mediapipe as mp
import numpy as np


class ExtractorLandmarks:
    """Detecta la mano y extrae un vector normalizado de 63 características (21 puntos x 3 ejes)."""

    def __init__(self, max_manos: int = 1, min_confianza_deteccion: float = 0.7, min_confianza_seguimiento: float = 0.6):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_manos,
            min_detection_confidence=min_confianza_deteccion,
            min_tracking_confidence=min_confianza_seguimiento,
        )

    def procesar_frame(self, frame_bgr: np.ndarray, dibujar: bool = True):
        """
        Procesa un frame BGR de OpenCV.
        Retorna:
            - frame_salida: imagen con el esqueleto dibujado (si dibujar=True)
            - vector_caracteristicas: np.ndarray de forma (63,) o None si no hay mano
            - mano_detectada: bool
        """
        frame_salida = frame_bgr.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        resultados = self.hands.process(frame_rgb)

        if not resultados.multi_hand_landmarks:
            return frame_salida, None, False

        hand_landmarks = resultados.multi_hand_landmarks[0]
        handedness = resultados.multi_handedness[0].classification[0].label  # "Left" o "Right"

        if dibujar:
            self.mp_drawing.draw_landmarks(
                frame_salida,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

        vector = self._normalizar_landmarks(hand_landmarks, handedness)
        return frame_salida, vector, True

    def _normalizar_landmarks(self, hand_landmarks, handedness: str) -> np.ndarray:
        """Convierte los 21 landmarks a coordenadas relativas a la muñeca y normalizadas por escala."""
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

        # Trasladar origen a la muñeca (punto 0)
        muneca = coords[0].copy()
        coords -= muneca

        # Estandarizar lateralidad (espejar eje X si es mano izquierda para unificar criterios)
        if handedness == "Left":
            coords[:, 0] = -coords[:, 0]

        # Escalar por la distancia máxima desde la muñeca (invarianza a distancia de cámara)
        distancias = np.linalg.norm(coords, axis=1)
        max_dist = np.max(distancias)
        if max_dist > 1e-6:
            coords /= max_dist

        return coords.flatten()

    def cerrar(self):
        self.hands.close()