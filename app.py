import anthropic
import cv2
import base64
import json
import os
import tempfile
import datetime
from PIL import Image
import io
import gradio as gr
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Colores ScoutAI ───────────────────────────────────────────────────────────
VERDE       = colors.HexColor('#1a3a2a')
VERDE_CLARO = colors.HexColor('#a3e635')
VERDE_MED   = colors.HexColor('#162210')
GRIS        = colors.HexColor('#7a9a7a')
NARANJA     = colors.HexColor('#ff9f43')
AMARILLO    = colors.HexColor('#f5c842')
ROJO        = colors.HexColor('#ff6b6b')
BLANCO      = colors.white
NEGRO       = colors.HexColor('#1c1c1c')

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
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": frame}})
    content.append({"type": "text", "text": build_prompt(nombre_habilidad, criterios, observaciones, duracion)})
    response = client.messages.create(model="claude-sonnet-4-6", max_tokens=2000, messages=[{"role": "user", "content": content}])
    raw = response.content[0].text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Generador de PDF ──────────────────────────────────────────────────────────
def generar_pdf(resultado, nombre_habilidad):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        tmp.name, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    st_titulo    = ParagraphStyle('titulo',    fontSize=22, textColor=VERDE_CLARO, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=4)
    st_subtitulo = ParagraphStyle('sub',       fontSize=13, textColor=GRIS,        fontName='Helvetica',       alignment=TA_CENTER, spaceAfter=16)
    st_seccion   = ParagraphStyle('seccion',   fontSize=12, textColor=VERDE_CLARO, fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    st_normal    = ParagraphStyle('normal',    fontSize=10, textColor=NEGRO,       fontName='Helvetica',       alignment=TA_JUSTIFY, spaceAfter=6, leading=14)
    st_italic    = ParagraphStyle('italic',    fontSize=10, textColor=NEGRO,       fontName='Helvetica-Oblique', alignment=TA_JUSTIFY, spaceAfter=6, leading=14)
    st_small     = ParagraphStyle('small',     fontSize=8,  textColor=GRIS,        fontName='Helvetica',       alignment=TA_CENTER)
    st_bold      = ParagraphStyle('bold',      fontSize=10, textColor=NEGRO,       fontName='Helvetica-Bold',  spaceAfter=4)

    score = resultado.get("puntuacion_global", 0)
    nivel = resultado.get("nivel", "")

    color_score = VERDE_CLARO if score >= 8 else AMARILLO if score >= 6 else NARANJA if score >= 4 else ROJO
    color_nivel = {
        "Principiante": ROJO, "En desarrollo": NARANJA,
        "Intermedio": AMARILLO, "Avanzado": VERDE_CLARO, "Élite": colors.HexColor('#00d4ff')
    }.get(nivel, VERDE_CLARO)

    fecha = datetime.datetime.now().strftime("%d/%m/%Y")
    elements = []

    # ── Encabezado ─────────────────────────────────────────────────────────
    elements.append(Paragraph("⚽ ScoutAI", st_titulo))
    elements.append(Paragraph("Informe de Análisis de Rendimiento", st_subtitulo))
    elements.append(Paragraph(f"{nombre_habilidad.replace('⚽','').replace('🎯','').replace('💨','').replace('🏃','').replace('✈️','').strip()} · {fecha}", st_subtitulo))
    elements.append(HRFlowable(width="100%", thickness=2, color=VERDE_CLARO, spaceAfter=12))

    # ── Puntuación global ──────────────────────────────────────────────────
    score_data = [[
        Paragraph(f"<font size='36' color='#{color_score.hexval()[2:]}'><b>{score}</b></font>", ParagraphStyle('s', alignment=TA_CENTER)),
        Paragraph(f"<font size='11' color='#7a9a7a'>Puntuación Global</font><br/><font size='14' color='#{color_nivel.hexval()[2:]}'><b>{nivel}</b></font>", ParagraphStyle('n', alignment=TA_LEFT, leading=18)),
    ]]
    score_table = Table(score_data, colWidths=[4*cm, 12*cm])
    score_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), VERDE_MED),
        ('ROUNDEDCORNERS', [8]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 12),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(resultado.get("resumen", ""), st_normal))
    elements.append(Spacer(1, 8))

    # ── Fortalezas y áreas ────────────────────────────────────────────────
    fort_items = "".join(f"<bullet>•</bullet>{f}<br/>" for f in resultado.get("fortalezas", []))
    area_items = "".join(f"<bullet>•</bullet>{a}<br/>" for a in resultado.get("areas_mejora", []))
    fa_data = [[
        [Paragraph("<b>💪 Fortalezas</b>", ParagraphStyle('fh', fontSize=11, textColor=VERDE_CLARO, fontName='Helvetica-Bold', spaceAfter=6)),
         Paragraph(fort_items, ParagraphStyle('fi', fontSize=9, textColor=NEGRO, leading=14))],
        [Paragraph("<b>🎯 Áreas a Mejorar</b>", ParagraphStyle('ah', fontSize=11, textColor=NARANJA, fontName='Helvetica-Bold', spaceAfter=6)),
         Paragraph(area_items, ParagraphStyle('ai', fontSize=9, textColor=NEGRO, leading=14))],
    ]]
    fa_table = Table(fa_data, colWidths=[8.5*cm, 8.5*cm])
    fa_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#0d2a1a')),
        ('BACKGROUND', (1,0), (1,0), colors.HexColor('#2a1a0d')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 10),
        ('ROUNDEDCORNERS', [6]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2a4a30')),
    ]))
    elements.append(fa_table)
    elements.append(Spacer(1, 12))

    # ── Evaluación por criterio ────────────────────────────────────────────
    elements.append(Paragraph("📊 Evaluación por Criterio", st_seccion))
    for c in resultado.get("criterios_detalle", []):
        nota = c.get("nota", 5)
        color_b = VERDE_CLARO if nota >= 8 else AMARILLO if nota >= 6 else NARANJA if nota >= 4 else ROJO
        bar_filled = int((nota / 10) * 100)
        bar_data = [[
            Paragraph(f"<b>{c['criterio']}</b>", ParagraphStyle('cb', fontSize=9, fontName='Helvetica-Bold', textColor=NEGRO)),
            Paragraph(f"<font color='#{color_b.hexval()[2:]}'><b>{nota}/10</b></font>", ParagraphStyle('cn', fontSize=9, fontName='Helvetica-Bold', alignment=TA_CENTER)),
        ]]
        bar_table = Table(bar_data, colWidths=[13*cm, 3*cm])
        bar_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('PADDING', (0,0), (-1,-1), 4)]))
        elements.append(bar_table)
        # Barra de progreso
        prog_data = [['',' ']]
        prog_table = Table(prog_data, colWidths=[bar_filled * 0.16 * cm + 0.01, (100 - bar_filled) * 0.16 * cm + 0.01])
        prog_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), color_b),
            ('BACKGROUND', (1,0), (1,0), colors.HexColor('#2a4a30')),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [color_b, colors.HexColor('#2a4a30')]),
            ('LINEABOVE', (0,0), (-1,-1), 0, colors.white),
            ('LINEBELOW', (0,0), (-1,-1), 0, colors.white),
        ]))
        # Barra simple con rectángulos
        barra_data = [['', '']]
        w_llena = max(bar_filled * 0.155, 0.1) * cm
        w_vacia = max((100 - bar_filled) * 0.155, 0.1) * cm
        barra = Table(barra_data, colWidths=[w_llena, w_vacia], rowHeights=[0.3*cm])
        barra.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), color_b),
            ('BACKGROUND', (1,0), (1,0), colors.HexColor('#2a4a30')),
        ]))
        elements.append(barra)
        elements.append(Paragraph(c.get("comentario", ""), ParagraphStyle('cc', fontSize=8, textColor=GRIS, spaceAfter=8, leading=11)))

    elements.append(Spacer(1, 8))

    # ── Consejo del entrenador ────────────────────────────────────────────
    elements.append(Paragraph("🧑‍💼 Consejo del Entrenador", st_seccion))
    consejo_data = [[Paragraph(f'"{resultado.get("consejo_entrenador", "")}"', st_italic)]]
    consejo_table = Table(consejo_data, colWidths=[17*cm])
    consejo_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0d2a1a')),
        ('PADDING', (0,0), (-1,-1), 12),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#a3e63550')),
    ]))
    elements.append(consejo_table)
    elements.append(Spacer(1, 12))

    # ── Plan de entrenamiento ─────────────────────────────────────────────
    elements.append(Paragraph("🏋️ Plan de Entrenamiento", st_seccion))
    for i, ej in enumerate(resultado.get("plan_ejercicios", []), 1):
        ej_data = [[
            Paragraph(f"<font color='#a3e635'><b>{i}</b></font>", ParagraphStyle('en', fontSize=14, fontName='Helvetica-Bold', alignment=TA_CENTER)),
            [
                Paragraph(f"<b>{ej.get('nombre','')}</b>", ParagraphStyle('enombre', fontSize=10, fontName='Helvetica-Bold', textColor=NEGRO, spaceAfter=3)),
                Paragraph(f"<font color='#a3e635'>Objetivo:</font> {ej.get('objetivo','')}", ParagraphStyle('eobj', fontSize=9, textColor=NEGRO, spaceAfter=3)),
                Paragraph(ej.get('descripcion',''), ParagraphStyle('edesc', fontSize=9, textColor=NEGRO, leading=13, spaceAfter=4)),
                Paragraph(f"<font color='#f5c842'>Volumen:</font> {ej.get('repeticiones','')}  &nbsp;&nbsp; <font color='#f5c842'>Frecuencia:</font> {ej.get('frecuencia','')}", ParagraphStyle('evol', fontSize=9, textColor=NEGRO)),
            ]
        ]]
        ej_table = Table(ej_data, colWidths=[1.5*cm, 15.5*cm])
        ej_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#0f1e16')),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#2a4a30')),
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('PADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(ej_table)
        elements.append(Spacer(1, 6))

    # ── Pie de página ─────────────────────────────────────────────────────
    elements.append(Spacer(1, 16))
    elements.append(HRFlowable(width="100%", thickness=1, color=GRIS))
    elements.append(Paragraph("ScoutAI · Análisis generado por Inteligencia Artificial · Complementar con criterio de entrenador real", st_small))

    doc.build(elements)
    return tmp.name

# ── Reporte HTML ──────────────────────────────────────────────────────────────
def generar_reporte_html(resultado, nombre_habilidad):
    score = resultado["puntuacion_global"]
    nivel = resultado["nivel"]
    color_score = "#a3e635" if score >= 8 else "#f5c842" if score >= 6 else "#ff9f43" if score >= 4 else "#ff6b6b"
    color_nivel = {"Principiante": "#ff6b6b", "En desarrollo": "#ff9f43", "Intermedio": "#f5c842", "Avanzado": "#a3e635", "Élite": "#00d4ff"}.get(nivel, "#a3e635")

    barras_html = ""
    for c in resultado.get("criterios_detalle", []):
        nota = c.get("nota", 5)
        color_barra = "#a3e635" if nota >= 8 else "#f5c842" if nota >= 6 else "#ff9f43" if nota >= 4 else "#ff6b6b"
        barras_html += f"""<div style="margin-bottom:14px">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                <span style="font-size:13px;color:#e8f0e2">{c['criterio']}</span>
                <span style="font-size:13px;font-weight:700;color:{color_barra}">{nota}/10</span>
            </div>
            <div style="height:8px;background:#2a4a30;border-radius:4px;overflow:hidden">
                <div style="height:100%;width:{nota*10}%;background:{color_barra};border-radius:4px"></div>
            </div>
            <div style="font-size:11px;color:#7a9a7a;margin-top:3px">{c.get('comentario','')}</div>
        </div>"""

    fortalezas_html = "".join(f'<div style="font-size:13px;color:#e8f0e2;padding:6px 10px;border-left:3px solid #a3e635;margin-bottom:6px;background:#0d2a1a;border-radius:0 6px 6px 0">{f}</div>' for f in resultado.get("fortalezas", []))
    areas_html = "".join(f'<div style="font-size:13px;color:#e8f0e2;padding:6px 10px;border-left:3px solid #ff9f43;margin-bottom:6px;background:#2a1a0d;border-radius:0 6px 6px 0">{a}</div>' for a in resultado.get("areas_mejora", []))

    ejercicios_html = ""
    for i, ej in enumerate(resultado.get("plan_ejercicios", []), 1):
        ejercicios_html += f"""<div style="background:#0f1e16;border:1px solid #2a4a30;border-radius:10px;padding:14px;margin-bottom:10px">
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

    return f"""<div style="font-family:'Segoe UI',sans-serif;background:#0f1e16;color:#e8f0e2;padding:24px;border-radius:16px">
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
                <div style="font-size:13px;font-weight:700;color:#a3e635;margin-bottom:10px">💪 Fortalezas</div>{fortalezas_html}
            </div>
            <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:16px">
                <div style="font-size:13px;font-weight:700;color:#ff9f43;margin-bottom:10px">🎯 A mejorar</div>{areas_html}
            </div>
        </div>
        <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:20px;margin-bottom:16px">
            <div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:16px">📊 Evaluación por Criterio</div>{barras_html}
        </div>
        <div style="background:linear-gradient(135deg,#0d2a1a,#0f1e16);border:1px solid #a3e63550;border-radius:12px;padding:18px;margin-bottom:16px">
            <div style="font-size:13px;font-weight:700;color:#a3e635;margin-bottom:10px">🧑‍💼 Consejo del Entrenador</div>
            <p style="font-size:13px;color:#e8f0e2;margin:0;line-height:1.7;font-style:italic">"{resultado.get('consejo_entrenador','')}"</p>
        </div>
        <div style="background:#162210;border:1px solid #2a4a30;border-radius:12px;padding:20px;margin-bottom:16px">
            <div style="font-size:15px;font-weight:800;color:#ffffff;margin-bottom:4px">🏋️ Plan de Entrenamiento</div>
            <div style="font-size:12px;color:#7a9a7a;margin-bottom:14px">Ejercicios personalizados según el análisis</div>{ejercicios_html}
        </div>
        <div style="text-align:center;font-size:11px;color:#7a9a7a;margin-top:8px">ScoutAI · Análisis generado por Inteligencia Artificial · Complementar con criterio de entrenador real</div>
    </div>"""

# ── Estado global para guardar el último resultado ────────────────────────────
ultimo_resultado = {"data": None, "habilidad": None}

# ── Función principal ─────────────────────────────────────────────────────────
def procesar_video(video_path, habilidad_seleccionada, observaciones):
    global ultimo_resultado
    if video_path is None:
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ Por favor sube un video antes de analizar.</div>", gr.update(visible=False)
    if not habilidad_seleccionada:
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ Por favor selecciona una habilidad a analizar.</div>", gr.update(visible=False)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key.startswith("sk-ant-"):
        return "<div style='color:#ff6b6b;padding:20px'>⚠️ API Key no configurada.</div>", gr.update(visible=False)
    try:
        frames, duracion = extraer_frames(video_path, num_frames=4)
        info_habilidad = HABILIDADES[habilidad_seleccionada]
        criterios = info_habilidad["criterios"]
        resultado = analizar_con_ia(frames, habilidad_seleccionada, criterios, observaciones or "", duracion)
        ultimo_resultado["data"] = resultado
        ultimo_resultado["habilidad"] = habilidad_seleccionada
        html = generar_reporte_html(resultado, habilidad_seleccionada)
        return html, gr.update(visible=True)
    except json.JSONDecodeError:
        return "<div style='color:#ff6b6b;padding:20px'>❌ Error al procesar la respuesta de la IA. Intenta de nuevo.</div>", gr.update(visible=False)
    except Exception as e:
        return f"<div style='color:#ff6b6b;padding:20px'>❌ Error: {str(e)}</div>", gr.update(visible=False)

def descargar_pdf():
    global ultimo_resultado
    if not ultimo_resultado["data"]:
        return None
    return generar_pdf(ultimo_resultado["data"], ultimo_resultado["habilidad"])

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
            habilidad = gr.Dropdown(choices=list(HABILIDADES.keys()), label="Habilidad a evaluar", info="Selecciona qué aspecto técnico deseas analizar", value=None)
            guia_box = gr.HTML("")
            video_input = gr.Video(label="Video del jugador", sources=["upload"])
            gr.HTML("""<div style='background:#162210;border:1px solid #2a4a30;border-radius:8px;padding:12px;margin:8px 0;font-size:12px;color:#7a9a7a'>
                📹 <strong style='color:#a3e635'>Requisitos del video:</strong><br>
                · Duración: <strong style='color:#e8f0e2'>5 a 30 segundos</strong><br>
                · Formato: MP4, MOV o AVI · Peso máximo: 50 MB<br>
                · El jugador debe ser visible en todo momento
            </div>""")
            observaciones = gr.Textbox(label="Observaciones adicionales (opcional)", placeholder="Ej: Jugador de 17 años, dominante con pie derecho...", lines=3)
            btn_analizar = gr.Button("⚡ Analizar con IA", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.HTML("<div style='color:#a3e635;font-weight:700;font-size:14px;margin-bottom:8px'>📊 Informe de análisis</div>")
            reporte = gr.HTML(value="""<div style='background:#162210;border:1px dashed #2a4a30;border-radius:12px;padding:40px;text-align:center;color:#7a9a7a;font-family:Segoe UI,sans-serif'>
                <div style='font-size:40px;margin-bottom:12px'>🎬</div>
                <div style='font-size:15px;font-weight:600;color:#e8f0e2;margin-bottom:6px'>Listo para analizar</div>
                <div style='font-size:13px'>Selecciona una habilidad, sube el video<br>y presiona <strong style='color:#a3e635'>Analizar con IA</strong></div>
            </div>""")
            btn_pdf = gr.DownloadButton(
                label="📄 Descargar Informe en PDF",
                visible=False,
                value=descargar_pdf,
                variant="secondary",
            )

    def actualizar_guia(h):
        if not h:
            return ""
        guia = HABILIDADES[h]["guia"]
        return f"""<div style='background:#0d2a1a;border:1px solid #a3e63540;border-radius:8px;padding:12px;margin:8px 0;font-size:12px'>
            <strong style='color:#a3e635'>📹 Cómo grabar este clip:</strong><br>
            <span style='color:#e8f0e2'>{guia}</span>
        </div>"""

    habilidad.change(fn=actualizar_guia, inputs=[habilidad], outputs=[guia_box])
    btn_analizar.click(fn=procesar_video, inputs=[video_input, habilidad, observaciones], outputs=[reporte, btn_pdf])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
