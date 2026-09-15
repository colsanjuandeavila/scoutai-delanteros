import anthropic
import cv2
import base64
import json
import os
from PIL import Image
import io
import gradio as gr

# ── Definición de habilidades ─────────────────────────────────────────────────
HABILIDADES = {
    "⚽ Remate y Definición": {
        "criterios": [
            "Posición del cuerpo al rematar",
            "Superficie de contacto con el balón",
            "Postura de la pierna de apoyo",
            "Equilibrio y seguimiento (follow-through)",
            "Precisión y potencia generada"
        ],
        "guia": "Video lateral o diagonal desde atrás. Debe capturar la carrera de aproximación, el golpeo y el seguimiento del balón."
    },
    "🎯 Control y Primer Toque": {
        "criterios": [
            "Lectura anticipada del balón",
            "Superficie usada para el control",
            "Orientación del primer toque",
            "Postura corporal al recibir",
            "Rapidez de ejecución bajo presión"
        ],
        "guia": "Video frontal o lateral. Debe mostrar la llegada del balón y los dos primeros toques."
    },
    "💨 Regate 1v1": {
        "criterios": [
            "Lectura del defensor",
            "Uso de fintas y amagues",
            "Control del balón en movimiento",
            "Explosividad tras el regate",
            "Decisión de qué lado atacar"
        ],
        "guia": "Plano abierto lateral. Debe verse el encaramiento al defensor, la finta y la aceleración posterior."
    },
    "🏃 Movimiento sin Balón": {
        "criterios": [
            "Lectura del espacio disponible",
            "Timing del desmarque",
            "Variedad de movimientos (diagonal, en profundidad)",
            "Generación de espacio para compañeros",
            "Posicionamiento respecto a la línea defensiva"
        ],
        "guia": "Plano abierto desde tribuna o altura. Debe mostrar los movimientos del delantero durante una jugada."
    },
    "✈️ Juego Aéreo": {
        "criterios": [
            "Timing del salto",
            "Posición del cuello en el cabezazo",
            "Dirección y potencia del remate",
            "Anticipación al defensor",
            "Uso correcto de los brazos"
        ],
        "guia": "Plano lateral o diagonal. Debe capturar la carrera previa, el salto y el momento del remate de cabeza."
    }
}

# ── Extracción de frames ──────────────────────────────────────────────────────
def extraer_frames(video_path, num_frames=4):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video. Verifica que sea un archivo válido (MP4, MOV, AVI).")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duracion = total_frames / fps if fps > 0 else 0

    if total_frames == 0:
        raise ValueError("El video parece estar vacío o corrupto.")

    posiciones = [int(total_frames * p) for p in [0.2, 0.4, 0.6, 0.8]]
    posiciones = [min(p, total_frames - 1) for p in posiciones]

    frames_b64 = []
    for pos in posiciones[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        max_dim = 1280
        if max(img.width, img.height) > max_dim:
            ratio = max_dim / max(img.width, img.height)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames_b64.append(b64)

    cap.release()

    if not frames_b64:
        raise ValueError("No se pudieron extraer frames del video.")

    return frames_b64, round(duracion, 1)

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(nombre_habilidad, criterios, observaciones, duracion):
    return f"""Eres un entrenador de fútbol de élite con 20+ años de experiencia trabajando con delanteros de alto rendimiento. Has trabajado en academias profesionales europeas y latinoamericanas.

Se te proporcionan 4 frames extraídos de un video de {duracion} segundos de un delantero de fútbol.
Los frames están distribuidos a lo largo del video (20%, 40%, 60% y 80% del clip).

HABILIDAD A EVALUAR: "{nombre_habilidad}"

CRITERIOS DE EVALUACIÓN:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(criterios))}

OBSERVACIONES DEL EVALUADOR: {observaciones if observaciones.strip() else "Ninguna."}

INSTRUCCIÓN IMPORTANTE: Analiza los frames como si fueran una secuencia del movimiento del jugador.
Infiere el movimiento y la técnica a partir de las posiciones corporales visibles.
Si el video no muestra claramente la habilidad solicitada, indícalo en el resumen pero igual proporciona
el análisis técnico más completo posible con lo que puedas observar.

Responde ÚNICAMENTE con JSON válido, sin texto adicional, sin backticks:
{{
  "puntuacion_global": <número 1.0-10.0>,
  "nivel": <"Principiante" | "En desarrollo" | "Intermedio" | "Avanzado" | "Élite">,
  "resumen": "<2-3 oraciones sobre el rendimiento general>",
  "fortalezas": ["<fortaleza 1>", "<fortaleza 2>", "<fortaleza 3>"],
  "areas_mejora": ["<área 1>", "<área 2>", "<área 3>"],
  "criterios_detalle": [
    {{
      "criterio": "<nombre criterio>",
      "nota": <1-10>,
      "comentario": "<observación específica>"
    }}
  ],
  "plan_ejercicios": [
    {{
      "nombre": "<nombre>",
      "objetivo": "<qué entrena>",
      "descripcion": "<cómo ejecutarlo en 2-3 oraciones>",
      "repeticiones": "<volumen>",
      "frecuencia": "<días/semana>"
    }}
  ],
  "consejo_entrenador": "<párrafo directo al jugador en segunda persona, motivador y técnico>"
}}"""

# ── Llamada a Claude ──────────────────────────────────────────────────────────
def analizar_con_ia(frames_b64, nombre_habilidad, criterios, observaciones, duracion):
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    content = []
    for frame in frames_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}
        })
    content.append({
        "type": "text",
        "text": build_prompt(nombre_habilidad, criterios, observaciones, duracion)
    })
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": content}]
    )
    raw = response.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Reporte HTML ──────────────────────────────────────────────────────────────
def generar_reporte_html(resultado, nombre_habilidad):
    score = resultado["puntuacion_global"]
    nivel = resultado["nivel"]

    color_score = "#a3e635" if score >= 8 else "#f5c842" if score >= 6 else "#ff9f43" if score >= 4 else "#ff6b6b"
    color_nivel = {
        "Principiante": "#ff6b6b", "En desarrollo": "#ff9f43",
        "Intermedio": "#f5c842", "Avanzado": "#a3e635", "Élite": "#00d4ff"
    }.get(nivel, "#a3e635")

    barras_html = ""
    for c in resultado.get("criterios_detalle", []):
        nota = c.get("nota", 5)
        color_barra = "#a3e635" if nota >= 8 else "#f5c842" if nota >= 6 else "#ff9f43" if nota >= 4 else "#ff6b6b"
        barras_html += f"""
        <div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:13px;color:#e8f0e2">{c['criterio']}</span>
                <span style="font-size:13px;font-weight:700;color:{color_barra}">{nota}/10</span>
            </div>
            <div style="height:8px;background:#2a4a30;border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{nota*10}%;background:{color_barra};border-radius:4px"></div>
            </div>
            <div style="font-size:11px;color:#7a9a7a;margin-top:3px">{c.get('comentario','')}</div>
        </div>"""

    fortalezas_html = "".join(
        f'<div style="font-size:13px;color:#e8f0e2;padding:6px 10px;border-left:3px solid #a3e635;margin-bottom:6px;background:#0d2a1a;border-radius:0 6px 6px 0">{f}</div>'
        for f in resultado.get("fortalezas", [])
    )
    areas_html = "".join(
        f'<div style="font-size:13px;color:#e8f0e2;padding:6px 10px;border-left:3px solid #ff9f43;margin-bottom:6px;background:#2a1a0d;border-radius:0 6px 6px 0">{a}</div>'
        for a in resultado.get("areas_mejora", [])
    )

    ejercicios_html = ""
    for i, ej in enumerate(resultado.get("plan_ejercicios", []), 1):
        ejercicios_html += f"""
        <div style="background:#0f1e16;border:1px solid #2a4a30;border-radius:10px;padding:14px;margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="background:#6aab1a;color:#0f1e16;width:24px;height:24px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:12px;font-weight:900">{i}</span>
                <span style="font-weight:700;color:#ffffff;font-size:14px">{ej.get('nombre','')}</span>
            </div>
            <div style="font-size:12px;color:#a3e635;margin-bottom:4px">🎯 {ej.get('objetivo','')}</div>
            <div style="font-size:13px;color:#e8f0e2;line-height:1.6;margin-bottom:8px">{ej.get('descripcion','')}</div>
            <div style="display:flex;gap:20px">
                <div><span style="font-size:11px;color:#7a9a7a">Volumen: </span><span style="font-size:12px;color:#f5c842;font-weight:600">{ej.get('repeticiones','')}</span></div>
                <div><span style="font-size:11px;color:#7a9a7a">Frecuencia: </span><span style="font-size:12px;color:#f5c842;font-weight:600">{ej.get('frecuencia','')}</span></div>
            </div>
        </div>"""

    return f"""
    <div style="font-family:'Segoe UI',sans-serif;background:#0f1e16;color:#e8f0e2;padding:24px;border-radius:16px">
        <div style="background:linear-gradient(135deg,#0f2a1a,#1a3a25);border-radius:12px;padding:20px;margin-bottom:20px;border:1px solid #2a4a30">
            <div style="font-size:22px;font-weight:900;color:#ffffff;margin-bottom:4px">⚽ ScoutAI — Informe de Análisis</div>
            <div style="font-size:14px;color:#a3e635">{nombre_habilidad}</div>
        </div>
        <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:20px;margin-bottom:16px;text-align:center">
            <div style="font-size:56px;font-weight:900;color:{color_score};line-height:1">{score}</div>
            <div style="font-size:14px;color:#7a9a7a;margin-bottom:10px">/ 10</div>
            <span style="background:{color_nivel}20;color:{color_nivel};padding:4px 16px;border-radius:20px;font-size:13px;font-weight:700;border:1px solid {color_nivel}40">{nivel}</span>
            <p style="font-size:13px;color:#7a9a7a;margin:14px 0 0;line-height:1.6;text-align:left">{resultado.get('resumen','')}</p>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
            <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:16px">
                <div style="font-size:13px;font-weight:700;color:#a3e635;margin-bottom:10px">💪 Fortalezas</div>
                {fortalezas_html}
            </div>
            <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:16px">
                <div style="font-size:13px;font-weight:700;color:#ff9f43;margin-bottom:10px">🎯 A mejorar</div>
                {areas_html}
            </div>
        </div>
        <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:20px;margin-bottom:16px">
            <div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:16px">📊 Evaluación por Criterio</div>
            {barras_html}
        </div>
        <div style="background:linear-gradient(135deg,#0d2a1a,#0f1e16);border:1px solid #a3e63550;border-radius:12px;padding:18px;margin-bottom:16px">
            <div style="font-size:13px;font-weight:700;color:#a3e635;margin-bottom:10px">🧑‍💼 Consejo del Entrenador</div>
            <p style="font-size:13px;color:#e8f0e2;margin:0;line-height:1.7;font-style:italic">"{resultado.get('consejo_entrenador','')}"</p>
        </div>
        <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:20px;margin-bottom:16px">
            <div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:4px">🏋️ Plan de Entrenamiento</div>
            <div style="font-size:12px;color:#7a9a7a;margin-bottom:14px">Ejercicios personalizados según el análisis</div>
            {ejercicios_html}
        </div>
        <div style="text-align:center;font-size:11px;color:#7a9a7a;margin-top:8px">
            ScoutAI · Análisis generado por Inteligencia Artificial · Complementar con criterio de entrenador real
        </div>
    </div>"""

# ── Función principal Gradio ──────────────────────────────────────────────────
def procesar_video(video_path, habilidad_seleccionada, observaciones):
    if video_path is None:
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ Por favor sube un video antes de analizar.</div>"
    if not habilidad_seleccionada:
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ Por favor selecciona una habilidad a analizar.</div>"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key.startswith("sk-ant-"):
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ API Key no configurada. Contacta al administrador.</div>"
    try:
        frames, duracion = extraer_frames(video_path, num_frames=4)
        info_habilidad = HABILIDADES[habilidad_seleccionada]
        criterios = info_habilidad["criterios"]
        resultado = analizar_con_ia(frames, habilidad_seleccionada, criterios, observaciones or "", duracion)
        return generar_reporte_html(resultado, habilidad_seleccionada)
    except json.JSONDecodeError:
        return "<div style='color:#ff6b6b;padding:20px'>❌ Error al procesar la respuesta de la IA. Intenta de nuevo.</div>"
    except Exception as e:
        return f"<div style='color:#ff6b6b;padding:20px'>❌ Error: {str(e)}</div>"

# ── Interfaz Gradio ───────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0f1e16 !important; }
.gradio-container { max-width: 960px !important; margin: 0 auto !important; }
.gr-button-primary { background: #a3e635 !important; color: #0f1e16 !important; font-weight: 800 !important; border: none !important; }
.gr-button-primary:hover { background: #6aab1a !important; }
label { color: #a3e635 !important; font-weight: 600 !important; }
.gr-input, .gr-dropdown, .gr-textarea { background: #162210 !important; border: 1px solid #2a4a30 !important; color: #e8f0e2 !important; }
"""

with gr.Blocks(css=CSS, title="ScoutAI - Analizador de Delanteros") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:24px 0 16px;font-family:Segoe UI,sans-serif'>
        <div style='font-size:42px'>⚽</div>
        <h1 style='font-size:28px;font-weight:900;color:#a3e635;margin:8px 0 4px'>ScoutAI</h1>
        <p style='color:#7a9a7a;font-size:14px;margin:0'>Analizador de Rendimiento para Delanteros · Inteligencia Artificial</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div style='color:#a3e635;font-weight:700;font-size:14px;margin-bottom:8px'>📋 Configuración del análisis</div>")

            habilidad = gr.Dropdown(
                choices=list(HABILIDADES.keys()),
                label="Habilidad a evaluar",
                info="Selecciona qué aspecto técnico deseas analizar",
                value=None
            )
            guia_box = gr.HTML("")

            video_input = gr.Video(label="Video del jugador", sources=["upload"])

            gr.HTML("""
            <div style='background:#162210;border:1px solid #2a4a30;border-radius:8px;padding:12px;margin:8px 0;font-size:12px;color:#7a9a7a'>
                📹 <strong style='color:#a3e635'>Requisitos del video:</strong><br>
                · Duración: <strong style='color:#e8f0e2'>5 a 30 segundos</strong><br>
                · Formato: MP4, MOV o AVI<br>
                · Peso máximo: 50 MB<br>
                · El jugador debe ser visible en todo momento
            </div>
            """)

            observaciones = gr.Textbox(
                label="Observaciones adicionales (opcional)",
                placeholder="Ej: Jugador de 17 años, dominante con pie derecho...",
                lines=3
            )
            btn_analizar = gr.Button("⚡ Analizar con IA", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.HTML("<div style='color:#a3e635;font-weight:700;font-size:14px;margin-bottom:8px'>📊 Informe de análisis</div>")
            reporte = gr.HTML(value="""
            <div style='background:#162210;border:1px dashed #2a4a30;border-radius:12px;padding:40px;text-align:center;color:#7a9a7a;font-family:Segoe UI,sans-serif'>
                <div style='font-size:40px;margin-bottom:12px'>🎬</div>
                <div style='font-size:15px;font-weight:600;color:#e8f0e2;margin-bottom:6px'>Listo para analizar</div>
                <div style='font-size:13px'>Selecciona una habilidad, sube el video<br>y presiona <strong style='color:#a3e635'>Analizar con IA</strong></div>
            </div>
            """)

    def actualizar_guia(h):
        if not h:
            return ""
        guia = HABILIDADES[h]["guia"]
        return f"""<div style='background:#0d2a1a;border:1px solid #a3e63540;border-radius:8px;padding:12px;margin:8px 0;font-size:12px'>
            <strong style='color:#a3e635'>📹 Cómo grabar este clip:</strong><br>
            <span style='color:#e8f0e2'>{guia}</span>
        </div>"""

    habilidad.change(fn=actualizar_guia, inputs=[habilidad], outputs=[guia_box])
    btn_analizar.click(fn=procesar_video, inputs=[video_input, habilidad, observaciones], outputs=[reporte])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
