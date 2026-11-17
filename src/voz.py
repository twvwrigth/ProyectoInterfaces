import queue
import threading
import pyttsx3


class SintetizadorVoz:
    """Síntesis de voz asíncrona en español para retroalimentación multimodal."""

    def __init__(self, velocidad: int = 165):
        self.velocidad = velocidad
        self.activo = True
        self._cola = queue.Queue()
        self._ejecutando = True
        self._hilo = threading.Thread(target=self._bucle_voz, daemon=True)
        self._hilo.start()

    def _bucle_voz(self):
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            motor = pyttsx3.init()
            motor.setProperty("rate", self.velocidad)
            for voz in motor.getProperty("voices"):
                if "spanish" in voz.name.lower() or "es-" in voz.id.lower() or "es_" in voz.id.lower():
                    motor.setProperty("voice", voz.id)
                    break
        except Exception:
            motor = None

        while self._ejecutando:
            try:
                texto = self._cola.get(timeout=0.3)
            except queue.Empty:
                continue

            if texto is None:
                break

            if self.activo and motor is not None:
                try:
                    motor.say(texto)
                    motor.runAndWait()
                except Exception:
                    pass

    def hablar(self, texto: str, limpiar_pendientes: bool = True):
        if not self.activo or not texto:
            return
        if limpiar_pendientes:
            while not self._cola.empty():
                try:
                    self._cola.get_nowait()
                except queue.Empty:
                    break
        self._cola.put(texto)

    def alternar_silencio(self) -> bool:
        self.activo = not self.activo
        return self.activo

    def cerrar(self):
        self._ejecutando = False
        self._cola.put(None)