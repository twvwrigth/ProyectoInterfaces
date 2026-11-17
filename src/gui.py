from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from src.landmarks import ExtractorLandmarks
from src.reconocedor import ReconocedorLSM
from src.tutor import TutorLSM
from src.voz import SintetizadorVoz

ROOT_DIR = Path(__file__).resolve().parent.parent


class TutorLSMApp:
    """Interfaz gráfica multimodal del Tutor de Alfabeto en Lengua de Señas Mexicana."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Tutor Interactivo de Lengua de Señas Mexicana (LSM)")
        self.root.geometry("1120x680")
        self.root.configure(bg="#161822")
        self.root.resizable(False, False)

        # Módulos del sistema
        self.extractor = ExtractorLandmarks()
        self.reconocedor = ReconocedorLSM(str(ROOT_DIR / "data" / "modelo_lsm.pkl"))
        self.tutor = TutorLSM(tiempo_confirmacion=1.4, umbral_confianza=0.68)
        self.voz = SintetizadorVoz()

        # Cámara
        self.cap = cv2.VideoCapture(0)
        self.camara_activa = self.cap.isOpened()

        self._letra_mostrada_ref = None
        self._img_ref_tk = None
        self._img_cam_tk = None

        self._construir_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._al_cerrar)

        if not self.reconocedor.modelo_cargado:
            messagebox.showwarning(
                "Modelo no encontrado",
                "No se encontró 'data/modelo_lsm.pkl'.\n\n"
                "Ejecuta primero:\n"
                "1. python scripts/capturar_dataset.py (al menos 2 letras)\n"
                "2. python scripts/entrenar.py",
            )
        else:
            self.voz.hablar(
                f"Bienvenido al tutor de lengua de señas mexicana. Comencemos con la letra {self.tutor.letra_objetivo}."
            )

        self._actualizar_frame()

    def _construir_ui(self):
        header = tk.Frame(self.root, bg="#1f2232", height=60)
        header.pack(fill="x", side="top")

        tk.Label(
            header,
            text="Tutor Inteligente de Alfabeto LSM",
            font=("Segoe UI", 16, "bold"),
            fg="#f8fafc",
            bg="#1f2232",
        ).pack(side="left", padx=20, pady=12)

        controles_modo = tk.Frame(header, bg="#1f2232")
        controles_modo.pack(side="right", padx=20)

        self.btn_modo_alfabeto = tk.Button(
            controles_modo,
            text="Modo Alfabeto",
            font=("Segoe UI", 10, "bold"),
            bg="#38bdf8",
            fg="#0f172a",
            relief="flat",
            padx=12,
            pady=5,
            cursor="hand2",
            command=lambda: self._cambiar_modo("alfabeto"),
        )
        self.btn_modo_alfabeto.pack(side="left", padx=5)

        self.btn_modo_palabras = tk.Button(
            controles_modo,
            text="Modo Deletreo",
            font=("Segoe UI", 10, "bold"),
            bg="#334155",
            fg="#f8fafc",
            relief="flat",
            padx=12,
            pady=5,
            cursor="hand2",
            command=lambda: self._cambiar_modo("palabras"),
        )
        self.btn_modo_palabras.pack(side="left", padx=5)

        main_frame = tk.Frame(self.root, bg="#161822")
        main_frame.pack(fill="both", expand=True, padx=20, pady=15)

        # --- PANEL IZQUIERDO: CÁMARA Y DETECCIÓN ---
        left_panel = tk.Frame(main_frame, bg="#1f2232", bd=0, highlightthickness=1, highlightbackground="#2e3348")
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.lbl_video = tk.Label(left_panel, bg="#0f111a", width=640, height=460)
        self.lbl_video.pack(padx=12, pady=(12, 8))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "LSM.Horizontal.TProgressbar",
            troughcolor="#0f111a",
            background="#4ade80",
            bordercolor="#1f2232",
            lightcolor="#4ade80",
            darkcolor="#22c55e",
        )
        self.barra_progreso = ttk.Progressbar(
            left_panel,
            style="LSM.Horizontal.TProgressbar",
            orient="horizontal",
            length=616,
            mode="determinate",
            maximum=100,
        )
        self.barra_progreso.pack(pady=(0, 8))

        self.lbl_feedback = tk.Label(
            left_panel,
            text="Iniciando sistema de visión...",
            font=("Segoe UI", 11, "bold"),
            fg="#38bdf8",
            bg="#1f2232",
            wraplength=610,
            justify="center",
        )
        self.lbl_feedback.pack(pady=(2, 10), padx=10)

        # --- PANEL DERECHO: OBJETIVO, REFERENCIA Y MÉTRICAS ---
        right_panel = tk.Frame(main_frame, bg="#1f2232", width=390, highlightthickness=1, highlightbackground="#2e3348")
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)

        stats_frame = tk.Frame(right_panel, bg="#262a3d")
        stats_frame.pack(fill="x", padx=15, pady=(15, 10))

        self.lbl_puntos = tk.Label(
            stats_frame, text="Puntos: 0", font=("Segoe UI", 11, "bold"), fg="#fbbf24", bg="#262a3d"
        )
        self.lbl_puntos.pack(side="left", padx=15, pady=8)

        self.lbl_racha = tk.Label(
            stats_frame, text="Racha: 0", font=("Segoe UI", 11, "bold"), fg="#4ade80", bg="#262a3d"
        )
        self.lbl_racha.pack(side="right", padx=15, pady=8)

        self.lbl_palabra = tk.Label(
            right_panel,
            text="Progreso: Abecedario LSM",
            font=("Segoe UI", 11, "bold"),
            fg="#94a3b8",
            bg="#1f2232",
        )
        self.lbl_palabra.pack(pady=(4, 0))

        self.lbl_objetivo = tk.Label(
            right_panel,
            text="A",
            font=("Segoe UI", 48, "bold"),
            fg="#4ade80",
            bg="#1f2232",
        )
        self.lbl_objetivo.pack(pady=(0, 4))

        self.lbl_imagen_ref = tk.Label(right_panel, bg="#0f111a", width=220, height=220)
        self.lbl_imagen_ref.pack(pady=4)

        self.lbl_consejo = tk.Label(
            right_panel,
            text="",
            font=("Segoe UI", 9),
            fg="#cbd5e1",
            bg="#1f2232",
            wraplength=350,
            justify="center",
        )
        self.lbl_consejo.pack(pady=8, padx=15)

        btn_frame = tk.Frame(right_panel, bg="#1f2232")
        btn_frame.pack(side="bottom", fill="x", padx=15, pady=15)

        tk.Button(
            btn_frame,
            text="Escuchar Pista",
            font=("Segoe UI", 10, "bold"),
            bg="#3b82f6",
            fg="#ffffff",
            relief="flat",
            pady=7,
            cursor="hand2",
            command=self._repetir_pista,
        ).pack(fill="x", pady=(0, 6))

        sub_btns = tk.Frame(btn_frame, bg="#1f2232")
        sub_btns.pack(fill="x")

        tk.Button(
            sub_btns,
            text="Saltar Letra",
            font=("Segoe UI", 9, "bold"),
            bg="#475569",
            fg="#f8fafc",
            relief="flat",
            pady=6,
            cursor="hand2",
            command=self._saltar_letra,
        ).pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_voz = tk.Button(
            sub_btns,
            text="Voz: ON",
            font=("Segoe UI", 9, "bold"),
            bg="#10b981",
            fg="#0f172a",
            relief="flat",
            pady=6,
            cursor="hand2",
            command=self._toggle_voz,
        )
        self.btn_voz.pack(side="right", fill="x", expand=True, padx=(4, 0))

    def _cargar_imagen_referencia(self, letra: str):
        """Carga assets/referencias/<LETRA>.png|.jpg o genera una tarjeta visual si no existe."""
        if self._letra_mostrada_ref == letra:
            return
        self._letra_mostrada_ref = letra

        dir_ref = ROOT_DIR / "assets" / "referencias"
        ruta_png = dir_ref / f"{letra}.png"
        ruta_jpg = dir_ref / f"{letra}.jpg"
        ruta_encontrada = ruta_png if ruta_png.exists() else (ruta_jpg if ruta_jpg.exists() else None)

        if ruta_encontrada:
            img = Image.open(ruta_encontrada).convert("RGB").resize((220, 220), Image.Resampling.LANCZOS)
        else:
            img = Image.new("RGB", (220, 220), color=(24, 28, 44))
            draw = ImageDraw.Draw(img)
            draw.rectangle([6, 6, 213, 213], outline=(56, 189, 248), width=2)
            draw.text((18, 16), "GUIA VISUAL LSM", fill=(148, 163, 184))
            try:
                fuente_grande = ImageFont.truetype("segoeui.ttf", 72)
            except Exception:
                fuente_grande = ImageFont.load_default()
            draw.text((82, 65), letra, fill=(74, 222, 128), font=fuente_grande)
            draw.text((22, 175), f"Agrega {letra}.png en assets", fill=(100, 116, 139))

        self._img_ref_tk = ImageTk.PhotoImage(img)
        self.lbl_imagen_ref.configure(image=self._img_ref_tk)

    def _cambiar_modo(self, modo: str):
        self.tutor.cambiar_modo(modo)
        self.reconocedor.limpiar_historial()
        if modo == "alfabeto":
            self.btn_modo_alfabeto.configure(bg="#38bdf8", fg="#0f172a")
            self.btn_modo_palabras.configure(bg="#334155", fg="#f8fafc")
        else:
            self.btn_modo_palabras.configure(bg="#38bdf8", fg="#0f172a")
            self.btn_modo_alfabeto.configure(bg="#334155", fg="#f8fafc")
        self.voz.hablar(f"Modo {modo}. Haz la letra {self.tutor.letra_objetivo}.")

    def _repetir_pista(self):
        letra = self.tutor.letra_objetivo
        consejo = self.tutor.obtener_consejo(letra)
        self.voz.hablar(f"Letra {letra}. {consejo}")

    def _saltar_letra(self):
        self.tutor.racha = 0
        self.tutor.siguiente_objetivo()
        self.reconocedor.limpiar_historial()
        self.voz.hablar(f"Siguiente letra: {self.tutor.letra_objetivo}.")

    def _toggle_voz(self):
        activo = self.voz.alternar_silencio()
        if activo:
            self.btn_voz.configure(text="Voz: ON", bg="#10b981", fg="#0f172a")
            self.voz.hablar("Voz activada.")
        else:
            self.btn_voz.configure(text="Voz: OFF", bg="#ef4444", fg="#ffffff")

    def _actualizar_frame(self):
        if self.camara_activa:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_anotado, vector, mano_detectada = self.extractor.procesar_frame(frame, dibujar=True)
                letra_pred, confianza = self.reconocedor.predecir(vector)

                resultado = self.tutor.actualizar(letra_pred, confianza, mano_detectada)

                if mano_detectada and letra_pred:
                    color_hud = (74, 222, 128) if letra_pred == self.tutor.letra_objetivo else (56, 189, 248)
                    cv2.rectangle(frame_anotado, (12, 12), (255, 55), (15, 23, 42), -1)
                    cv2.putText(
                        frame_anotado,
                        f"Detectado: {letra_pred} ({int(confianza * 100)}%)",
                        (22, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_hud,
                        2,
                    )

                self.barra_progreso["value"] = self.tutor.progreso * 100
                self.lbl_feedback.configure(text=resultado["mensaje"])

                if resultado["estado"] in ("manteniendo", "completado"):
                    self.lbl_feedback.configure(fg="#4ade80")
                elif resultado["estado"] == "incorrecto":
                    self.lbl_feedback.configure(fg="#fbbf24")
                else:
                    self.lbl_feedback.configure(fg="#38bdf8")

                if resultado["hablar"]:
                    self.voz.hablar(resultado["hablar"])

                objetivo = self.tutor.letra_objetivo
                self.lbl_objetivo.configure(text=objetivo)
                self.lbl_consejo.configure(text=self.tutor.obtener_consejo(objetivo))
                self.lbl_puntos.configure(text=f"Puntos: {self.tutor.puntos}")
                self.lbl_racha.configure(text=f"Racha: {self.tutor.racha}")

                if self.tutor.modo == "alfabeto":
                    self.lbl_palabra.configure(
                        text=f"Alfabeto LSM ({self.tutor.indice_letra + 1}/{len(self.tutor.LETRAS_LSM)})"
                    )
                else:
                    palabra_visual = " ".join(
                        f"[{c}]" if i == self.tutor.indice_palabra else c
                        for i, c in enumerate(self.tutor.palabra_actual)
                    )
                    self.lbl_palabra.configure(text=f"Palabra: {palabra_visual}")

                self._cargar_imagen_referencia(objetivo)

                frame_rgb = cv2.cvtColor(frame_anotado, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb).resize((640, 460), Image.Resampling.BILINEAR)
                self._img_cam_tk = ImageTk.PhotoImage(img_pil)
                self.lbl_video.configure(image=self._img_cam_tk)
        else:
            self.lbl_feedback.configure(text="No se pudo abrir la cámara (índice 0).", fg="#ef4444")

        self.root.after(15, self._actualizar_frame)

    def _al_cerrar(self):
        if self.camara_activa:
            self.cap.release()
        self.extractor.cerrar()
        self.voz.cerrar()
        self.root.destroy()


def iniciar_app():
    root = tk.Tk()
    TutorLSMApp(root)
    root.mainloop()