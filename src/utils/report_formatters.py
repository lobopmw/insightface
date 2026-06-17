from __future__ import annotations

import io
import textwrap
from datetime import datetime
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

TITLE_COLOR = colors.HexColor("#17324D")
SUBTITLE_COLOR = colors.HexColor("#617285")
TEXT_COLOR = colors.HexColor("#243746")
MUTED_TEXT_COLOR = colors.HexColor("#6E7C8B")
BORDER_COLOR = colors.HexColor("#D9E2EC")
CARD_FILL = colors.HexColor("#F7FAFC")
PRIORITY_LOW_FILL = colors.HexColor("#EDF7F1")
PRIORITY_MEDIUM_FILL = colors.HexColor("#FFF6E8")
PRIORITY_HIGH_FILL = colors.HexColor("#FDEEEE")
WHITE = colors.white

BEHAVIOR_PDF_COLORS = {
    "Atento": "#60A5FA",
    "Distraido": "#F28C52",
    "Distraído": "#F28C52",
    "Perguntando": "#1D4ED8",
    "Escrevendo": "#2F855A",
    "Dormindo": "#8B5FBF",
    "Agitado": "#D97706",
    "Em Pé": "#94A3B8",
}

TEMPORAL_SEGMENT_COLORS = {
    "Início da aula": "#7CB4FF",
    "Meio da aula": "#4F7FEA",
    "Final da aula": "#274690",
}


def format_date_br(value) -> str:
    if value is None or value == "":
        return ""
    return pd.to_datetime(value).strftime("%d-%m-%Y")


def format_duration_human(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds or 0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}min {seconds}s"
    if minutes:
        return f"{minutes}min {seconds}s"
    return f"{seconds}s"


def format_duration_minutes_label(total_seconds: float) -> str:
    total_minutes = float(total_seconds or 0) / 60.0
    return f"{total_minutes:.1f} min"


def _classify_distribution_style(active_day_percentage: float, max_daily_share_percentage: float) -> str:
    if active_day_percentage >= 60 and max_daily_share_percentage <= 35:
        return "estável ao longo do período"
    if active_day_percentage < 35 or max_daily_share_percentage >= 50:
        return "com maior concentração em momentos específicos"
    return "com oscilações moderadas"


def _get_behavior_row(summary: pd.DataFrame, behavior_name: str):
    if summary.empty:
        return None
    row = summary[summary["behavior"] == behavior_name]
    if row.empty:
        return None
    return row.iloc[0]


def _get_predominant_behavior_row(summary: pd.DataFrame):
    if summary.empty:
        return None
    ordered = summary.sort_values(
        ["duration_percentage", "occurrence_percentage", "records", "behavior"],
        ascending=[False, False, False, True],
    )
    return ordered.iloc[0]


def _behavior_color(behavior: str) -> str:
    return BEHAVIOR_PDF_COLORS.get(str(behavior), "#6B7280")


def _build_behavior_distribution_rows(summary: pd.DataFrame) -> list[dict]:
    if summary.empty:
        return []
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "label": str(row["behavior"]),
                "share": float(row["duration_percentage"]),
                "minutes": float(row["duration_minutes"]),
                "color": _behavior_color(str(row["behavior"])),
            }
        )
    return rows


def _build_management_balance_rows(summary: pd.DataFrame) -> list[dict]:
    if summary.empty:
        return []

    behavior_share = {
        str(row["behavior"]): float(row["duration_percentage"])
        for _, row in summary.iterrows()
    }
    rows = [
        {"label": "Engajamento", "value": behavior_share.get("Atento", 0.0) + behavior_share.get("Escrevendo", 0.0)},
        {"label": "Participação", "value": behavior_share.get("Perguntando", 0.0)},
        {
            "label": "Sinais de atenção",
            "value": (
                behavior_share.get("Distraído", 0.0)
                + behavior_share.get("Distraido", 0.0)
                + behavior_share.get("Dormindo", 0.0)
                + behavior_share.get("Agitado", 0.0)
            ),
        },
        {"label": "Movimento", "value": behavior_share.get("Em Pé", 0.0)},
    ]
    colors_by_group = {
        "Engajamento": "#3B82F6",
        "Participação": "#2563EB",
        "Sinais de atenção": "#F28C52",
        "Movimento": "#94A3B8",
    }
    return [
        {"label": row["label"], "value": round(float(row["value"]), 2), "color": colors_by_group[row["label"]]}
        for row in rows
        if float(row["value"]) > 0
    ]


def _build_temporal_segment_rows(report_data: dict) -> list[dict]:
    segment_distribution = report_data["session_segment_distribution"]
    if segment_distribution.empty:
        return []

    grouped = (
        segment_distribution.groupby("session_segment", as_index=False)["records"]
        .sum()
    )
    total_records = max(int(grouped["records"].sum()), 1)
    ordered_segments = ["Início da aula", "Meio da aula", "Final da aula"]
    rows = []
    for segment in ordered_segments:
        row = grouped[grouped["session_segment"] == segment]
        records = int(row.iloc[0]["records"]) if not row.empty else 0
        rows.append(
            {
                "label": segment,
                "records": records,
                "share": round(records / total_records * 100.0, 2),
                "color": TEMPORAL_SEGMENT_COLORS[segment],
            }
        )
    return rows


def _build_comparison_rows(report_data: dict) -> list[dict]:
    comparison = report_data["comparison"]
    summary = report_data["behavior_summary"]
    if comparison.empty or summary.empty or int(comparison["previous_records"].sum()) <= 0:
        return []

    focus_behaviors = summary["behavior"].head(3).tolist()
    rows = []
    for behavior in focus_behaviors:
        row = comparison[comparison["behavior"] == behavior]
        if row.empty:
            continue
        row = row.iloc[0]
        previous_records = int(row["previous_records"])
        current_records = int(row["current_records"])
        if previous_records <= 0:
            continue
        delta = current_records - previous_records
        delta_pct = (delta / previous_records) * 100.0
        if delta > 0:
            arrow = "↑"
            tone = "aumento"
            color = "#D97706"
        elif delta < 0:
            arrow = "↓"
            tone = "redução"
            color = "#2563EB"
        else:
            arrow = "→"
            tone = "estabilidade"
            color = "#64748B"
        rows.append(
            {
                "behavior": behavior,
                "current": current_records,
                "previous": previous_records,
                "delta_pct": abs(delta_pct),
                "arrow": arrow,
                "tone": tone,
                "color": color,
            }
        )
    return rows


def _build_observational_priority(report_data: dict) -> dict[str, str]:
    summary = report_data["behavior_summary"]
    temporal_rows = _build_temporal_segment_rows(report_data)
    if summary.empty:
        return {
            "label": "Baixa prioridade observacional",
            "reason": "O volume de dados disponível é reduzido, o que recomenda leitura conservadora neste momento.",
            "fill": PRIORITY_LOW_FILL,
            "accent": "#2F855A",
        }

    attentive = _get_behavior_row(summary, "Atento")
    distracted = _get_behavior_row(summary, "Distraído")
    if distracted is None:
        distracted = _get_behavior_row(summary, "Distraido")
    agitated = _get_behavior_row(summary, "Agitado")
    sleeping = _get_behavior_row(summary, "Dormindo")

    attentive_share = float(attentive["duration_percentage"]) if attentive is not None else 0.0
    attention_signal_share = 0.0
    for row in [distracted, agitated, sleeping]:
        if row is not None:
            attention_signal_share += float(row["duration_percentage"])

    top_temporal_share = max((float(row["share"]) for row in temporal_rows), default=0.0)

    if attention_signal_share >= 25.0 or (attention_signal_share >= 18.0 and attentive_share < 45.0):
        return {
            "label": "Atenção observacional elevada",
            "reason": "O conjunto de registros sugere presença relevante de comportamentos que pedem acompanhamento mais próximo em novas observações.",
            "fill": PRIORITY_HIGH_FILL,
            "accent": "#C53030",
        }
    if attention_signal_share >= 12.0 or top_temporal_share >= 50.0 or attentive_share < 60.0:
        return {
            "label": "Atenção observacional moderada",
            "reason": "Os dados indicam oscilações que merecem leitura contextualizada e acompanhamento pedagógico em coletas futuras.",
            "fill": PRIORITY_MEDIUM_FILL,
            "accent": "#C05621",
        }
    return {
        "label": "Baixa prioridade observacional",
        "reason": "O período sugere predominância de engajamento, sem sinais fortes de atenção adicional neste recorte.",
        "fill": PRIORITY_LOW_FILL,
        "accent": "#2F855A",
    }


def build_observational_summary(report_data: dict) -> str:
    summary = report_data["behavior_summary"]
    metrics = report_data["headline_metrics"]

    if summary.empty:
        return (
            "Não houve registros suficientes para compor uma síntese observacional do período selecionado."
        )

    top_behavior = _get_predominant_behavior_row(summary)
    secondary_summary = summary[summary["behavior"] != top_behavior["behavior"]]
    second_behavior = None if secondary_summary.empty else secondary_summary.sort_values(
        ["duration_percentage", "occurrence_percentage", "records", "behavior"],
        ascending=[False, False, False, True],
    ).iloc[0]
    temporal_rows = _build_temporal_segment_rows(report_data)
    text = (
        f"No período analisado, observou-se predomínio de {str(top_behavior['behavior']).lower()}. "
        f"Foram contabilizados {metrics['total_records']} episódios em {metrics['active_days']} dia(s), com "
        f"tempo total observado de {format_duration_human(metrics['total_duration_seconds'])}."
    )

    if second_behavior is not None and float(second_behavior["duration_percentage"]) >= 10.0:
        text += (
            f" Também apareceram momentos de {str(second_behavior['behavior']).lower()}, o que indica variações "
            "no modo de participação ao longo das aulas observadas."
        )

    if temporal_rows:
        top_segment = max(temporal_rows, key=lambda row: row["share"])
        text += (
            f" A maior concentração de registros ocorreu no {top_segment['label'].lower()}, "
            "o que ajuda a orientar o olhar pedagógico sobre esse trecho da aula."
        )

    text += (
        " Esses resultados devem ser lidos como apoio à análise pedagógica e sempre considerados à luz do contexto "
        "da atividade, da dinâmica da turma e das condições de observação."
    )
    return text


def generate_interpretive_summary(report_data: dict) -> str:
    summary = report_data["behavior_summary"]
    consistency = report_data["behavior_consistency"]
    if summary.empty:
        return "Não houve base suficiente para uma leitura interpretativa do período."

    top_behavior = _get_predominant_behavior_row(summary)
    top_behavior_name = str(top_behavior["behavior"]).lower()
    consistency_row = consistency[consistency["behavior"] == top_behavior["behavior"]]
    consistency_text = ""
    if not consistency_row.empty:
        row = consistency_row.iloc[0]
        consistency_text = (
            f"Esse padrão se manteve {_classify_distribution_style(float(row['active_day_percentage']), float(row['max_daily_share_percentage']))}."
        )

    distractions = []
    for name in ["Distraído", "Distraido", "Agitado", "Dormindo"]:
        row = _get_behavior_row(summary, name)
        if row is not None and float(row["duration_percentage"]) >= 8.0:
            distractions.append(str(name).lower())

    variation_text = ""
    if distractions:
        variation_text = (
            f" Também foram identificados momentos de {', '.join(dict.fromkeys(distractions))}, "
            "sugerindo oscilações de engajamento em partes do período."
        )

    parts = [f"O comportamento {top_behavior_name} predominou no período observado."]
    if consistency_text:
        parts.append(consistency_text)
    if variation_text:
        parts.append(variation_text.strip())
    parts.append("Os resultados devem ser interpretados como apoio à análise pedagógica, sem finalidade diagnóstica.")
    return " ".join(part for part in parts if part).strip()


def generate_temporal_distribution_summary(report_data: dict) -> str:
    rows = _build_temporal_segment_rows(report_data)
    segment_distribution = report_data["session_segment_distribution"]
    summary = report_data["behavior_summary"]
    if not rows or segment_distribution.empty:
        return "Não houve base temporal suficiente para uma síntese da aula."

    top_segment = max(rows, key=lambda row: row["share"])
    top_segment_behavior = (
        segment_distribution[segment_distribution["session_segment"] == top_segment["label"]]
        .sort_values(["records", "total_duration_seconds"], ascending=[False, False])
        .iloc[0]["behavior"]
    )
    predominant_row = _get_predominant_behavior_row(summary)
    predominant_behavior = None if predominant_row is None else str(predominant_row["behavior"])
    negative_behaviors = {"Distraído", "Distraido", "Agitado", "Dormindo"}

    if (
        predominant_behavior is not None
        and str(predominant_behavior).lower() == "atento"
        and str(top_segment_behavior) in negative_behaviors
    ):
        return (
            f"Embora o comportamento geral tenha sido {str(predominant_behavior).lower()}, "
            f"no {top_segment['label'].lower()} houve maior concentração de registros de "
            f"{str(top_segment_behavior).lower()}. "
            "Esse contraste sugere atenção pedagógica a esse trecho da aula e acompanhamento em novas observações."
        )

    if top_segment["share"] >= 45:
        return (
            f"Os registros se concentraram principalmente no {top_segment['label'].lower()}, "
            f"momento em que houve maior presença de {str(top_segment_behavior).lower()}. "
            "Esse padrão sugere observar com atenção a dinâmica didática desse trecho da aula."
        )
    return (
        "Os registros ficaram distribuídos de forma relativamente equilibrada entre os diferentes momentos da aula, "
        f"com leve destaque para o {top_segment['label'].lower()}. "
        "Essa distribuição sugere variação de engajamento ao longo da atividade."
    )


def generate_consistency_summary(report_data: dict) -> str:
    consistency = report_data["behavior_consistency"]
    summary = report_data["behavior_summary"]
    if consistency.empty or summary.empty:
        return "Não houve base suficiente para avaliar a consistência do comportamento no período."

    predominant_behavior = _get_predominant_behavior_row(summary)["behavior"]
    predominant_row = consistency[consistency["behavior"] == predominant_behavior].iloc[0]
    label = str(predominant_row["consistency_label"]).lower()
    if "regular" in label:
        return (
            f"O comportamento {str(predominant_behavior).lower()} apareceu com boa constância ao longo do período, "
            "o que sugere maior estabilidade no modo de participação."
        )
    if "epis" in label:
        return (
            f"Esse comportamento esteve presente ao longo do período, "
            "com maior concentração em momentos específicos."
        )
    return (
        f"O comportamento {str(predominant_behavior).lower()} manteve predominância geral, embora com oscilações "
        "ao longo das observações."
    )


def generate_observational_attention_points(report_data: dict) -> list[str]:
    summary = report_data["behavior_summary"]
    temporal_rows = _build_temporal_segment_rows(report_data)
    comparison_rows = _build_comparison_rows(report_data)
    points: list[str] = []

    if not summary.empty:
        distracted_row = _get_behavior_row(summary, "Distraído")
        if distracted_row is None:
            distracted_row = _get_behavior_row(summary, "Distraido")
        asking_row = _get_behavior_row(summary, "Perguntando")
        sleeping_row = _get_behavior_row(summary, "Dormindo")

        if distracted_row is not None and float(distracted_row["duration_percentage"]) >= 10.0:
            points.append("Presença de distração em parcela relevante do período observado.")
        if sleeping_row is not None and float(sleeping_row["duration_percentage"]) >= 6.0:
            points.append("Sinais de sonolência merecem leitura contextualizada da rotina e do tipo de atividade.")
        if asking_row is not None and float(asking_row["duration_percentage"]) >= 8.0:
            points.append("Há momentos de participação ativa que podem ser mobilizados em estratégias de engajamento.")

    if temporal_rows:
        top_segment = max(temporal_rows, key=lambda row: row["share"])
        points.append(f"A maior concentração de registros ocorreu no {top_segment['label'].lower()}.")

    if comparison_rows:
        top_change = max(comparison_rows, key=lambda row: row["delta_pct"])
        points.append(
            f"Na comparação com o período anterior, {str(top_change['behavior']).lower()} apresentou a variação mais perceptível."
        )

    if not points:
        points.append(
            "Os registros do período não indicaram concentrações suficientemente fortes para destacar novos pontos de atenção."
        )

    return points[:4]


def generate_previous_period_comparison(report_data: dict) -> list[str]:
    rows = _build_comparison_rows(report_data)
    if not rows:
        return ["Não houve registros suficientes no período anterior para uma comparação analítica consistente."]

    lines = []
    for row in rows:
        if row["tone"] == "estabilidade":
            lines.append(
                f"{row['arrow']} {row['behavior']}: manteve estabilidade em relação ao período anterior."
            )
        else:
            lines.append(
                f"{row['arrow']} {row['behavior']}: {row['tone']} de {row['delta_pct']:.1f}% em relação ao período anterior."
            )
    return lines


def generate_methodological_note() -> str:
    return (
        "O tempo acumulado apresentado no relatório corresponde a uma estimativa construída a partir da duração dos episódios observados. "
        "Essa medida pode variar em função da taxa de amostragem, de oclusões, de perdas momentâneas de detecção e das condições de captação. "
        "Por isso, deve ser compreendida como referência analítica e não como medição absoluta do comportamento."
    )


def build_limitations_text() -> str:
    return (
        "As informações deste relatório são observacionais e derivadas de reconhecimento facial, detecção de pose e regras geométricas. "
        "A plataforma pode ser influenciada por iluminação, enquadramento, movimentação coletiva, qualidade do fluxo de vídeo e contexto pedagógico. "
        "Os resultados não constituem diagnóstico e devem ser utilizados exclusivamente como apoio à análise pedagógica e acadêmica."
    )


def build_report_pdf(report_data: dict) -> bytes:
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    margin_x = 42
    usable_width = width - (margin_x * 2)
    page_top = height - 42
    page_bottom = 36
    cursor_y = page_top
    page_number = 1

    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=TEXT_COLOR,
    )
    small_style = ParagraphStyle(
        "Small",
        fontName="Helvetica",
        fontSize=8.5,
        leading=13,
        alignment=TA_LEFT,
        textColor=MUTED_TEXT_COLOR,
    )

    def draw_page_footer():
        pdf.setStrokeColor(BORDER_COLOR)
        pdf.setLineWidth(0.8)
        pdf.line(margin_x, page_bottom + 10, margin_x + usable_width, page_bottom + 10)
        pdf.setFont("Helvetica", 8)
        pdf.setFillColor(MUTED_TEXT_COLOR)
        pdf.drawString(margin_x, page_bottom - 2, "Plataforma de apoio observacional pedagógico")
        pdf.drawRightString(margin_x + usable_width, page_bottom - 2, f"Página {page_number}")

    def next_page():
        nonlocal cursor_y, page_number
        draw_page_footer()
        pdf.showPage()
        page_number += 1
        cursor_y = page_top

    def ensure_space(required_height: float):
        nonlocal cursor_y
        if cursor_y - required_height < page_bottom + 26:
            next_page()

    def ensure_section_space(min_total: float):
        ensure_space(min_total)

    def write_paragraph(text: str, style: ParagraphStyle = body_style, after: float = 8):
        nonlocal cursor_y
        safe_text = escape(str(text)).replace("\n", "<br/>")
        paragraph = Paragraph(safe_text, style)
        _, paragraph_height = paragraph.wrap(usable_width, page_top)
        ensure_space(paragraph_height + after)
        paragraph.drawOn(pdf, margin_x, cursor_y - paragraph_height)
        cursor_y -= paragraph_height + after

    def write_bullets(lines: list[str], font_size: int = 10, bullet_color=TEXT_COLOR):
        nonlocal cursor_y
        for line in lines:
            wrapped = textwrap.wrap(str(line), width=90) or [""]
            ensure_space(16 * len(wrapped) + 2)
            first = True
            for wrapped_line in wrapped:
                pdf.setFont("Helvetica", font_size)
                pdf.setFillColor(TEXT_COLOR)
                if first:
                    pdf.setFillColor(bullet_color)
                    pdf.drawString(margin_x, cursor_y, "•")
                    pdf.setFillColor(TEXT_COLOR)
                    pdf.drawString(margin_x + 12, cursor_y, wrapped_line)
                    first = False
                else:
                    pdf.drawString(margin_x + 12, cursor_y, wrapped_line)
                cursor_y -= 13
            cursor_y -= 2

    def write_section(title: str, min_following_space: float = 42):
        nonlocal cursor_y
        ensure_section_space(46 + min_following_space)
        pdf.setFont("Helvetica-Bold", 14.5)
        pdf.setFillColor(TITLE_COLOR)
        pdf.drawString(margin_x, cursor_y, title[:120])
        cursor_y -= 15
        pdf.setStrokeColor(BORDER_COLOR)
        pdf.setLineWidth(0.8)
        pdf.line(margin_x, cursor_y, margin_x + usable_width, cursor_y)
        cursor_y -= 28

    def draw_round_box(x: float, y: float, w: float, h: float, fill_color, stroke_color=BORDER_COLOR, radius: float = 10):
        pdf.setFillColor(fill_color)
        pdf.setStrokeColor(stroke_color)
        pdf.roundRect(x, y, w, h, radius, fill=1, stroke=1)

    def draw_header():
        nonlocal cursor_y
        title = "Relatório de Monitoramento Comportamental"
        student = report_data.get("student") or "Aluno não informado"
        period = f"{format_date_br(report_data['start_date'])} a {format_date_br(report_data['end_date'])}"
        emitted = datetime.now().strftime("%d-%m-%Y")

        ensure_space(108)
        pdf.setFont("Helvetica-Bold", 20)
        pdf.setFillColor(TITLE_COLOR)
        pdf.drawString(margin_x, cursor_y, title)
        cursor_y -= 22

        pdf.setFont("Helvetica", 10.5)
        pdf.setFillColor(SUBTITLE_COLOR)
        pdf.drawString(margin_x, cursor_y, f"{student}  |  {period}  |  {report_data['period_mode']}")
        cursor_y -= 14
        pdf.drawString(margin_x, cursor_y, f"Emitido em: {emitted}")
        cursor_y -= 18

        pdf.setStrokeColor(BORDER_COLOR)
        pdf.setLineWidth(1)
        pdf.line(margin_x, cursor_y, margin_x + usable_width, cursor_y)
        cursor_y -= 22

    def draw_summary_cards(metrics: dict, summary: pd.DataFrame):
        nonlocal cursor_y
        ensure_space(126)
        card_gap = 14
        card_width = (usable_width - (card_gap * 3)) / 4
        card_height = 90
        card_y = cursor_y - card_height

        predominant_row = _get_predominant_behavior_row(summary)
        predominant_behavior = str(
            predominant_row["behavior"] if predominant_row is not None else metrics["predominant_behavior"]
        )
        cards = [
            ("Comportamento predominante", predominant_behavior),
            ("Tempo total observado", format_duration_minutes_label(metrics["total_duration_seconds"])),
            ("Total de episódios", str(metrics["total_records"])),
            ("Dias com registros", str(metrics["active_days"])),
        ]

        for index, (title, value) in enumerate(cards):
            x = margin_x + index * (card_width + card_gap)
            draw_round_box(x, card_y, card_width, card_height, CARD_FILL)
            title_lines = textwrap.wrap(title, width=22)[:2] or [title]
            pdf.setFont("Helvetica-Bold", 7.6)
            pdf.setFillColor(MUTED_TEXT_COLOR)
            title_y = card_y + card_height - 16
            for line in title_lines:
                pdf.drawString(x + 10, title_y, line)
                title_y -= 9
            pdf.setFont("Helvetica-Bold", 14)
            pdf.setFillColor(TITLE_COLOR)
            value_text = value[:22]
            pdf.drawString(x + 10, card_y + 28, value_text)

        cursor_y = card_y - 24

    def draw_donut_chart(x: float, y_top: float, width_box: float, height_box: float, rows: list[dict]):
        if not rows:
            return y_top

        draw_round_box(x, y_top - height_box, width_box, height_box, WHITE)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColor(TITLE_COLOR)
        pdf.drawString(x + 12, y_top - 20, "Distribuição percentual por comportamento")

        center_x = x + 78
        center_y = y_top - 88
        radius = 46
        inner_radius = 24
        start_angle = 90
        total_share = sum(max(0.0, float(row["share"])) for row in rows) or 1.0
        for row in rows:
            extent = 360.0 * max(0.0, float(row["share"])) / total_share
            pdf.setFillColor(colors.HexColor(row["color"]))
            pdf.wedge(center_x - radius, center_y - radius, center_x + radius, center_y + radius, start_angle, extent, stroke=0, fill=1)
            start_angle += extent
        pdf.setFillColor(WHITE)
        pdf.circle(center_x, center_y, inner_radius, stroke=0, fill=1)
        pdf.setFillColor(TITLE_COLOR)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawCentredString(center_x, center_y + 3, "100%")
        pdf.setFont("Helvetica", 7.5)
        pdf.setFillColor(MUTED_TEXT_COLOR)
        pdf.drawCentredString(center_x, center_y - 10, "período")

        legend_x = x + 136
        legend_y = y_top - 46
        for row in rows[:5]:
            pdf.setFillColor(colors.HexColor(row["color"]))
            pdf.roundRect(legend_x, legend_y - 7, 8, 8, 2, fill=1, stroke=0)
            pdf.setFont("Helvetica", 8.5)
            pdf.setFillColor(TEXT_COLOR)
            pdf.drawString(legend_x + 14, legend_y - 1, str(row["label"])[:18])
            pdf.setFont("Helvetica-Bold", 8.5)
            pdf.drawRightString(x + width_box - 10, legend_y - 1, f"{float(row['share']):.1f}%")
            legend_y -= 16

        return y_top - height_box

    def draw_vertical_bars_chart(x: float, y_top: float, width_box: float, height_box: float, rows: list[dict]):
        if not rows:
            return y_top

        draw_round_box(x, y_top - height_box, width_box, height_box, WHITE)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColor(TITLE_COLOR)
        pdf.drawString(x + 12, y_top - 20, "Tempo acumulado por comportamento")

        chart_left = x + 16
        chart_bottom = y_top - height_box + 50
        chart_width = width_box - 32
        chart_height = 82
        max_minutes = max(float(row["minutes"]) for row in rows) or 1.0
        chart_rows = rows[:5]
        bar_width = 28
        gap = (chart_width - (len(chart_rows) * bar_width)) / max(len(chart_rows) + 1, 1)

        pdf.setStrokeColor(BORDER_COLOR)
        pdf.line(chart_left, chart_bottom, chart_left + chart_width, chart_bottom)

        for index, row in enumerate(chart_rows):
            bar_x = chart_left + gap + index * (bar_width + gap)
            fill_height = chart_height * (float(row["minutes"]) / max_minutes)
            pdf.setFillColor(colors.HexColor(row["color"]))
            pdf.roundRect(bar_x, chart_bottom, bar_width, fill_height, 3, fill=1, stroke=0)
            pdf.setFont("Helvetica-Bold", 8)
            pdf.setFillColor(TITLE_COLOR)
            pdf.drawCentredString(bar_x + bar_width / 2, chart_bottom + fill_height + 8, f"{float(row['minutes']):.1f} min")
            pdf.setFont("Helvetica", 6.8)
            label_lines = textwrap.wrap(str(row["label"]), width=12)[:2] or [str(row["label"])]
            for line_index, line in enumerate(label_lines):
                pdf.drawCentredString(bar_x + bar_width / 2, chart_bottom - 11 - (line_index * 8), line)

        return y_top - height_box

    def draw_temporal_chart(rows: list[dict]):
        nonlocal cursor_y
        ensure_space(226)
        box_height = 172
        draw_round_box(margin_x, cursor_y - box_height, usable_width, box_height, WHITE)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.setFillColor(TITLE_COLOR)
        pdf.drawString(margin_x + 12, cursor_y - 22, "Distribuição temporal da aula")

        chart_x = margin_x + 18
        chart_y = cursor_y - box_height + 30
        chart_width = usable_width - 36
        label_width = 72
        chart_height = 88
        bar_height = 16
        bar_gap = 14
        max_share = max(float(row["share"]) for row in rows) or 1.0

        pdf.setStrokeColor(BORDER_COLOR)
        axis_x = chart_x + label_width
        pdf.line(axis_x, chart_y + 4, axis_x, chart_y + chart_height + 8)

        for index, row in enumerate(rows):
            y = chart_y + chart_height - ((index + 1) * (bar_height + bar_gap))
            fill_width = (chart_width - label_width - 34) * (float(row["share"]) / max_share)
            pdf.setFont("Helvetica-Bold", 8.5)
            pdf.setFillColor(TITLE_COLOR)
            label = row["label"].replace(" da aula", "")
            pdf.drawString(chart_x, y + 4, label)

            pdf.setFillColor(colors.HexColor("#E9EEF5"))
            pdf.roundRect(axis_x + 8, y, chart_width - label_width - 34, bar_height, 4, fill=1, stroke=0)
            pdf.setFillColor(colors.HexColor(row["color"]))
            pdf.roundRect(axis_x + 8, y, fill_width, bar_height, 4, fill=1, stroke=0)
            pdf.setFont("Helvetica-Bold", 8.5)
            pdf.setFillColor(TITLE_COLOR)
            pdf.drawString(axis_x + 18 + fill_width, y + 4, f"{float(row['share']):.1f}%")

        cursor_y = cursor_y - box_height - 16

    def draw_comparison_cards(rows: list[dict]):
        nonlocal cursor_y
        if not rows:
            write_paragraph(
                "Não houve registros suficientes no período anterior para uma comparação analítica consistente."
            )
            return

        ensure_space(92 + (len(rows[:3]) * 48))
        for row in rows[:3]:
            card_height = 40
            card_y = cursor_y - card_height
            draw_round_box(margin_x, card_y, usable_width, card_height, CARD_FILL)
            pdf.setFont("Helvetica-Bold", 12)
            pdf.setFillColor(colors.HexColor(row["color"]))
            pdf.drawString(margin_x + 12, card_y + 15, row["arrow"])
            pdf.setFont("Helvetica-Bold", 10)
            pdf.setFillColor(TITLE_COLOR)
            pdf.drawString(margin_x + 28, card_y + 24, str(row["behavior"])[:24])
            pdf.setFont("Helvetica", 8.5)
            pdf.setFillColor(TEXT_COLOR)
            if row["tone"] == "estabilidade":
                detail = "manteve estabilidade em relação ao período anterior"
            else:
                detail = f"{row['tone']} de {row['delta_pct']:.1f}% no número de episódios"
            pdf.drawString(margin_x + 28, card_y + 11, detail[:78])
            cursor_y = card_y - 8

    def draw_priority_block(priority: dict[str, str]):
        nonlocal cursor_y
        ensure_space(92)
        block_height = 62
        block_y = cursor_y - block_height
        draw_round_box(margin_x, block_y, usable_width, block_height, priority["fill"])
        pdf.setFont("Helvetica-Bold", 11)
        pdf.setFillColor(colors.HexColor(priority["accent"]))
        pdf.drawString(margin_x + 12, block_y + 40, priority["label"])
        pdf.setFont("Helvetica", 9)
        pdf.setFillColor(TEXT_COLOR)
        pdf.drawString(margin_x + 12, block_y + 22, priority["reason"][:110])
        cursor_y = block_y - 16

    metrics = report_data["headline_metrics"]
    summary = report_data["behavior_summary"]
    behavior_rows = _build_behavior_distribution_rows(summary)
    temporal_rows = _build_temporal_segment_rows(report_data)
    comparison_rows = _build_comparison_rows(report_data)
    executive_insights = generate_observational_attention_points(report_data)
    observational_priority = _build_observational_priority(report_data)

    pdf.setTitle("Relatório de Monitoramento Comportamental")
    draw_header()
    draw_summary_cards(metrics, summary)

    write_section("Resumo executivo", min_following_space=62)
    write_paragraph(build_observational_summary(report_data), after=18)

    write_section("Leitura visual dos comportamentos", min_following_space=228)
    ensure_space(232)
    donut_width = (usable_width - 12) / 2
    chart_top = cursor_y
    chart_height = 182
    draw_donut_chart(margin_x, chart_top, donut_width, chart_height, behavior_rows)
    draw_vertical_bars_chart(margin_x + donut_width + 12, chart_top, donut_width, chart_height, behavior_rows)
    cursor_y = chart_top - chart_height - 22

    write_section("Análise interpretativa", min_following_space=84)
    write_paragraph(generate_interpretive_summary(report_data), after=12)
    write_paragraph(generate_consistency_summary(report_data), after=18)

    write_section("Distribuição temporal", min_following_space=232)
    if temporal_rows:
        draw_temporal_chart(temporal_rows)
        write_paragraph(generate_temporal_distribution_summary(report_data), after=18)
    else:
        write_paragraph("Não houve base suficiente para apresentar a distribuição temporal do período.", after=18)

    write_section("Sinais para acompanhamento pedagógico", min_following_space=72)
    write_bullets(executive_insights[:4], bullet_color=colors.HexColor("#2563EB"))
    cursor_y -= 14

    write_section("Indicação de prioridade observacional", min_following_space=82)
    draw_priority_block(observational_priority)

    write_section("Comparação com o período anterior", min_following_space=78)
    draw_comparison_cards(comparison_rows)

    cursor_y -= 8
    write_section("Nota metodológica", min_following_space=58)
    write_paragraph(generate_methodological_note(), style=small_style, after=14)

    write_section("Limitações", min_following_space=58)
    write_paragraph(build_limitations_text(), style=small_style, after=12)

    draw_page_footer()
    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
