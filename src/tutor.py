import random
import time


class TutorLSM:
    """Motor pedagógico que evalúa las señas del usuario y genera retroalimentación."""

    LETRAS_LSM = [
        "A", "B", "C", "D", "E", "F", "G",
        "H", "I", "L", "M", "N", "O", "P",
        "R", "S", "T", "U", "V", "W", "Y",
    ]

    PALABRAS_PRACTICA = ["HOLA", "LSM", "CASA", "AMOR", "MEXICO", "SOL", "VIDA", "MANO"]

    CONSEJOS_LSM = {
        "A": "Cierra el puño con los dedos juntos y apoya el pulgar recto al costado del índice.",
        "B": "Extiende los cuatro dedos juntos hacia arriba y dobla el pulgar sobre la palma.",
        "C": "Curva todos los dedos y el pulgar formando la silueta de una letra C.",
        "D": "Levanta solo el dedo índice recto; une las yemas del pulgar y los demás dedos en círculo.",
        "E": "Flexiona las puntas de los cuatro dedos sobre la palma y cruza el pulgar por debajo.",
        "F": "Une las yemas del índice y pulgar; mantén medio, anular y meñique estirados arriba.",
        "G": "Apunta el índice horizontalmente hacia un lado con el pulgar paralelo encima.",
        "H": "Extiende índice y medio juntos en horizontal hacia un lado; recoge los demás dedos.",
        "I": "Cierra el puño y levanta únicamente el dedo meñique bien estirado.",
        "L": "Extiende el índice hacia arriba y el pulgar hacia afuera formando un ángulo de 90°.",
        "M": "Dobla índice, medio y anular hacia abajo sobre el pulgar, con las puntas apuntando al suelo.",
        "N": "Dobla solo índice y medio hacia abajo sobre el pulgar; recoge anular y meñique.",
        "O": "Une las puntas de todos los dedos con la yema del pulgar formando un óvalo cerrado.",
        "P": "Como la letra K/V invertida: índice adelante, medio hacia abajo y pulgar entre ambos.",
        "R": "Levanta índice y medio estirados y cruza el dedo medio por detrás del índice.",
        "S": "Cierra el puño firmemente y cruza el pulgar por delante de los dedos doblados.",
        "T": "Cierra el puño y coloca el pulgar asomando entre el dedo índice y el dedo medio.",
        "U": "Levanta índice y medio completamente juntos y estirados hacia arriba.",
        "V": "Levanta índice y medio estirados y sepáralos formando una V.",
        "W": "Levanta índice, medio y anular estirados y separados; sujeta el meñique con el pulgar.",
        "Y": "Extiende el pulgar y el meñique hacia los lados manteniendo doblados los tres dedos centrales.",
    }

    def __init__(self, tiempo_confirmacion: float = 1.4, umbral_confianza: float = 0.68):
        self.tiempo_confirmacion = tiempo_confirmacion
        self.umbral_confianza = umbral_confianza

        self.modo = "alfabeto"  # "alfabeto" | "palabras"
        self.indice_letra = 0
        self.palabra_actual = self.PALABRAS_PRACTICA[0]
        self.indice_palabra = 0

        self.puntos = 0
        self.racha = 0
        self.completadas = 0

        self._inicio_acierto = None
        self._inicio_intento = time.time()
        self._pista_dada = False
        self.progreso = 0.0

    @property
    def letra_objetivo(self) -> str:
        if self.modo == "alfabeto":
            return self.LETRAS_LSM[self.indice_letra]
        return self.palabra_actual[self.indice_palabra]

    def obtener_consejo(self, letra: str = None) -> str:
        objetivo = letra or self.letra_objetivo
        return self.CONSEJOS_LSM.get(objetivo, "Coloca tu mano frente a la cámara con buena iluminación.")

    def cambiar_modo(self, nuevo_modo: str):
        self.modo = nuevo_modo
        self._reiniciar_temporizadores()
        if nuevo_modo == "alfabeto":
            self.indice_letra = 0
        else:
            self.palabra_actual = random.choice(self.PALABRAS_PRACTICA)
            self.indice_palabra = 0

    def siguiente_objetivo(self):
        self._reiniciar_temporizadores()
        if self.modo == "alfabeto":
            self.indice_letra = (self.indice_letra + 1) % len(self.LETRAS_LSM)
        else:
            self.indice_palabra += 1
            if self.indice_palabra >= len(self.palabra_actual):
                self.palabra_actual = random.choice(self.PALABRAS_PRACTICA)
                self.indice_palabra = 0

    def _reiniciar_temporizadores(self):
        self._inicio_acierto = None
        self._inicio_intento = time.time()
        self._pista_dada = False
        self.progreso = 0.0

    def actualizar(self, letra_detectada: str, confianza: float, mano_detectada: bool) -> dict:
        """
        Evalúa el estado actual de la seña frente al objetivo.
        Retorna un diccionario con:
            - estado: "sin_mano" | "incorrecto" | "manteniendo" | "completado"
            - mensaje: str para mostrar en pantalla
            - hablar: str | None (frase para sintetizador de voz si hubo evento clave)
        """
        ahora = time.time()
        objetivo = self.letra_objetivo

        if not mano_detectada or letra_detectada is None:
            self._inicio_acierto = None
            self.progreso = 0.0
            return {
                "estado": "sin_mano",
                "mensaje": f"Muestra tu mano haciendo la letra '{objetivo}'.",
                "hablar": None,
            }

        # ¿Coincide con la letra esperada?
        if letra_detectada == objetivo and confianza >= self.umbral_confianza:
            if self._inicio_acierto is None:
                self._inicio_acierto = ahora

            transcurrido = ahora - self._inicio_acierto
            self.progreso = min(1.0, transcurrido / self.tiempo_confirmacion)

            if self.progreso >= 1.0:
                self.puntos += 10 + (self.racha * 2)
                self.racha += 1
                self.completadas += 1
                letra_lograda = objetivo
                palabra_terminada = (
                    self.modo == "palabras" and self.indice_palabra == len(self.palabra_actual) - 1
                )
                palabra_completada_texto = self.palabra_actual

                self.siguiente_objetivo()
                nuevo_obj = self.letra_objetivo

                if palabra_terminada:
                    msg = f"¡Excelente! Completaste la palabra {palabra_completada_texto}. Siguiente: {nuevo_obj}."
                    voz = f"Excelente, completaste la palabra {palabra_completada_texto}. Ahora haz la letra {nuevo_obj}."
                else:
                    msg = f"¡Correcto! Letra {letra_lograda} dominada. Siguiente: {nuevo_obj}."
                    voz = f"Correcto, letra {letra_lograda}. Ahora haz la letra {nuevo_obj}."

                return {
                    "estado": "completado",
                    "mensaje": msg,
                    "hablar": voz,
                }

            return {
                "estado": "manteniendo",
                "mensaje": f"¡Bien! Mantén la seña '{objetivo}' firme ({int(self.progreso * 100)}%)...",
                "hablar": None,
            }

        # Seña incorrecta o baja confianza
        self._inicio_acierto = None
        self.progreso = 0.0

        # Si lleva más de 7 segundos intentando sin éxito, dar retroalimentación guiada
        if (ahora - self._inicio_intento) > 7.0 and not self._pista_dada:
            self._pista_dada = True
            self.racha = 0
            consejo = self.obtener_consejo(objetivo)
            return {
                "estado": "incorrecto",
                "mensaje": f"Detecto '{letra_detectada}'. Consejo para '{objetivo}': {consejo}",
                "hablar": f"Estás haciendo la {letra_detectada}. Para la {objetivo}: {consejo}",
            }

        return {
            "estado": "incorrecto",
            "mensaje": f"Detecto '{letra_detectada}' ({int(confianza * 100)}%). Busca la postura de '{objetivo}'.",
            "hablar": None,
        }