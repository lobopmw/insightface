from __future__ import annotations

import io
import textwrap
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph


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


def _classify_distribution_style(active_day_percentage: float, max_daily_share_percentage: float) -> str:
    if active_day_percentage >= 60 and max_daily_share_percentage <= 35:
        return "distribuição regular"
    if active_day_percentage < 35 or max_daily_share_percentage >= 50:
        return "distribuição concentrada em dias específicos"
    return "distribuição intermediária"


def _get_behavior_row(summary: pd.DataFrame, behavior_name: str):
    if summary.empty:
        return None
    row = summary[summary["behavior"] == behavior_name]
    if row.empty:
        return None
    return row.iloc[0]


def build_observational_summary(report_data: dict) -> str:
    summary = report_data["behavior_summary"]
    metrics = report_data["headline_metrics"]

    if summary.empty:
        return (
            "Não houve registros suficientes para o aluno e o período selecionados. "
            "O sistema depende de episódios observacionais previamente persistidos."
        )

    top_behavior = summary.iloc[0]
    second_behavior = summary.iloc[1] if len(summary) > 1 else None
    asking_row = _get_behavior_row(summary, "Perguntando")

    text = (
        f"No período analisado, o sistema registrou {metrics['total_records']} episódios observacionais em "
        f"{metrics['active_days']} dia(s), com duração acumulada estimada de "
        f"{format_duration_human(metrics['total_duration_seconds'])}. "
        f"Os dados indicam maior frequência registrada de ocorrências classificadas como "
        f"'{top_behavior['behavior']}', correspondendo a {top_behavior['occurrence_percentage']:.2f}% "
        f"dos registros e a {top_behavior['duration_minutes']:.2f} minuto(s) acumulados."
    )
    if second_behavior is not None:
        text += (
            f" Como ocorrência relativa subsequente, houve frequência registrada relevante de episódios "
            f"classificados como '{second_behavior['behavior']}', com {second_behavior['occurrence_percentage']:.2f}% dos registros."
        )
    if asking_row is not None and asking_row["behavior"] not in {top_behavior["behavior"], second_behavior["behavior"] if second_behavior is not None else ""} and float(asking_row["occurrence_percentage"]) >= 15.0:
        text += (
            f" Também houve recorrência registrada de ocorrências classificadas como 'Perguntando', "
            f"com {asking_row['occurrence_percentage']:.2f}% dos registros do período."
        )
    text += (
        " Trata-se de indicador observacional derivado de visão computacional. Os resultados devem ser interpretados "
        "à luz do contexto pedagógico da aula e das condições de captação."
    )
    return text


def generate_interpretive_summary(report_data: dict) -> str:
    summary = report_data["behavior_summary"]
    consistency = report_data["behavior_consistency"]

    if summary.empty:
        return "Não houve base suficiente para a leitura interpretativa do período."

    top_behavior = summary.iloc[0]
    top_consistency = consistency[consistency["behavior"] == top_behavior["behavior"]]
    if not top_consistency.empty:
        row = top_consistency.iloc[0]
        distribution_style = _classify_distribution_style(
            float(row["active_day_percentage"]),
            float(row["max_daily_share_percentage"]),
        )
        consistency_text = (
            f"Foram observados padrões compatíveis com '{top_behavior['behavior']}' em {distribution_style}, com recorrência registrada em "
            f"{int(row['days_with_occurrence'])} dia(s) do período."
        )
    else:
        consistency_text = (
            f"Houve maior frequência registrada de ocorrências classificadas como '{top_behavior['behavior']}' no período selecionado."
        )

    episodic_behaviors = []
    if not consistency.empty:
        episodic_behaviors = (
            consistency[consistency["consistency_label"] == "Episódica"]["behavior"].head(2).tolist()
        )

    episodic_text = ""
    if episodic_behaviors:
        names = ", ".join(f"'{name}'" for name in episodic_behaviors)
        episodic_text = (
            f" Também foram observados padrões compatíveis com {names} de forma mais episódica ou concentrada, "
            "o que recomenda uma leitura contextualizada das aulas correspondentes."
        )

    pedagogical_text = ""
    asking_row = _get_behavior_row(summary, "Perguntando")
    if (
        asking_row is not None
        and float(top_behavior["occurrence_percentage"]) >= 35.0
        and float(asking_row["occurrence_percentage"]) >= 12.0
    ):
        pedagogical_text = (
            " Os registros também sugerem alternância entre acompanhamento contínuo da aula e momentos de "
            "interação observacionalmente classificados pelo sistema, sempre devendo essa leitura ser situada "
            "no contexto pedagógico do período analisado."
        )

    return (
        f"A leitura interpretativa do período indica predominância de padrões observacionais compatíveis com "
        f"'{top_behavior['behavior']}' entre os registros produzidos pelo sistema. {consistency_text}{episodic_text} "
        f"{pedagogical_text}"
        "Esses resultados não devem ser analisados isoladamente, mas em conjunto com as "
        "condições de aula, a dinâmica pedagógica e as características de captação."
    )


def generate_temporal_distribution_summary(report_data: dict) -> str:
    segment_distribution = report_data["session_segment_distribution"]

    if segment_distribution.empty:
        return (
            "Não houve registros temporais suficientes para sintetizar a distribuição do período analisado."
        )

    overall = (
        segment_distribution.groupby("session_segment", as_index=False)["records"]
        .sum()
        .sort_values("records", ascending=False)
    )
    top_segment = overall.iloc[0]
    total_records = max(int(overall["records"].sum()), 1)
    top_segment_percentage = float(top_segment["records"]) / total_records * 100.0

    top_behavior_by_segment = (
        segment_distribution[segment_distribution["session_segment"] == top_segment["session_segment"]]
        .sort_values(["records", "total_duration_seconds"], ascending=[False, False])
        .iloc[0]
    )

    if top_segment_percentage >= 45:
        distribution_style = f"maior concentração de registros no intervalo '{top_segment['session_segment']}'"
        support_text = (
            "a leitura temporal sugere concentração relativa nesse trecho do período observado"
        )
    else:
        distribution_style = "variação dos padrões ao longo do período observado"
        support_text = (
            "a leitura temporal sugere distribuição mais equilibrada dos registros ao longo da aula"
        )

    return (
        f"Na distribuição temporal do período, {support_text}. "
        f"O segmento '{top_segment['session_segment']}' reuniu {top_segment_percentage:.2f}% dos registros, "
        f"o que indica {distribution_style}. "
        f"Neste recorte temporal, houve maior frequência registrada de ocorrências classificadas como "
        f"'{top_behavior_by_segment['behavior']}'. "
        "Essa distribuição deve ser analisada em conjunto com o planejamento pedagógico, "
        "com o tipo de atividade desenvolvida e com as condições de observação."
    )


def generate_consistency_summary(report_data: dict) -> str:
    consistency = report_data["behavior_consistency"]
    summary = report_data["behavior_summary"]

    if consistency.empty or summary.empty:
        return "Não houve base suficiente para avaliar a consistência comportamental no período."

    predominant_behavior = summary.iloc[0]["behavior"]
    predominant_row = consistency[consistency["behavior"] == predominant_behavior].iloc[0]

    if predominant_row["consistency_label"] == "Regular":
        main_text = (
            f"Foram observados padrões compatíveis com '{predominant_behavior}' de forma regular ao longo dos dias observados, "
            f"com recorrência registrada em {int(predominant_row['days_with_occurrence'])} dia(s)."
        )
    elif predominant_row["consistency_label"] == "Episódica":
        main_text = (
            f"Embora as ocorrências classificadas como '{predominant_behavior}' tenham predominado no agregado, sua recorrência registrada mostrou concentração "
            f"em parte dos dias observados, com presença em {int(predominant_row['days_with_occurrence'])} dia(s)."
        )
    else:
        main_text = (
            f"As ocorrências classificadas como '{predominant_behavior}' mantiveram predominância geral, com distribuição intermediária ao longo dos dias "
            f"do período analisado."
        )

    episodic = consistency[consistency["consistency_label"] == "Episódica"]["behavior"].tolist()
    episodic = [behavior for behavior in episodic if behavior != predominant_behavior][:2]
    if episodic:
        episodic_names = ", ".join("'" + behavior + "'" for behavior in episodic)
        episodic_verb = "apareceu" if len(episodic) == 1 else "apareceram"
        episodic_text = (
            f" Em contrapartida, {episodic_names} "
            f"{episodic_verb} de forma mais concentrada em dias específicos."
        )
    else:
        episodic_text = ""

    asking_text = ""
    asking_row = _get_behavior_row(summary, "Perguntando")
    if asking_row is not None and float(asking_row["occurrence_percentage"]) >= 12.0:
        asking_consistency = consistency[consistency["behavior"] == "Perguntando"]
        if not asking_consistency.empty:
            asking_consistency = asking_consistency.iloc[0]
            if asking_consistency["consistency_label"] == "Regular":
                asking_text = (
                    " Também houve recorrência registrada de ocorrências classificadas como 'Perguntando' em diferentes dias do período."
                )
            else:
                asking_text = (
                    " As ocorrências classificadas como 'Perguntando' também apareceram com relevância no agregado, "
                    "embora com distribuição menos regular ao longo dos dias."
                )

    return main_text + episodic_text + asking_text


def generate_observational_attention_points(report_data: dict) -> list[str]:
    points: list[str] = []
    peak_days = report_data["peak_days"]
    segment_distribution = report_data["session_segment_distribution"]
    consistency = report_data["behavior_consistency"]
    summary = report_data["behavior_summary"]

    if not peak_days.empty:
        top_peak = peak_days.sort_values(["records", "duration_minutes"], ascending=[False, False]).iloc[0]
        points.append(
            f"Foram observados picos de recorrência de '{top_peak['behavior']}' em "
            f"{format_date_br(top_peak['date'])}, sugerindo a verificação do contexto da aula correspondente."
        )

    if not segment_distribution.empty:
        top_segment = (
            segment_distribution.groupby("session_segment", as_index=False)["records"]
            .sum()
            .sort_values("records", ascending=False)
            .iloc[0]
        )
        points.append(
            f"O segmento temporal '{top_segment['session_segment']}' concentrou a maior frequência registrada no período."
        )

    if not consistency.empty:
        concentrated = consistency[consistency["consistency_label"] == "Episódica"]["behavior"].head(2).tolist()
        if concentrated:
            concentrated_names = ", ".join("'" + name + "'" for name in concentrated)
            points.append(
                f"Alguns indicadores observacionais, como {concentrated_names}, "
                "apareceram de forma concentrada em recortes específicos, recomendando leitura articulada com o planejamento pedagógico."
            )

    asking_row = _get_behavior_row(summary, "Perguntando")
    if asking_row is not None and float(asking_row["occurrence_percentage"]) >= 15.0:
        points.append(
            "Houve frequência registrada relevante de ocorrências classificadas como 'Perguntando', o que recomenda observação do contexto didático e da dinâmica de interação das aulas correspondentes."
        )

    if not points:
        points.append(
            "Não houve concentrações suficientemente destacadas para gerar pontos adicionais de atenção observacional."
        )

    return points


def generate_previous_period_comparison(report_data: dict) -> list[str]:
    comparison = report_data["comparison"]
    summary = report_data["behavior_summary"]

    if comparison.empty or summary.empty:
        return ["Não há base comparativa suficiente para o período imediatamente anterior."]

    valid_previous = comparison["previous_records"].sum()
    if int(valid_previous) <= 0:
        return ["Não houve registros suficientes no período imediatamente anterior para uma comparação analítica consistente."]

    main_behaviors = summary["behavior"].head(3).tolist()
    asking_row = _get_behavior_row(summary, "Perguntando")
    if asking_row is not None and "Perguntando" not in main_behaviors and float(asking_row["occurrence_percentage"]) >= 12.0:
        main_behaviors.append("Perguntando")
    lines: list[str] = []
    for behavior in main_behaviors:
        row = comparison[comparison["behavior"] == behavior]
        if row.empty:
            continue
        row = row.iloc[0]
        current_records = int(row["current_records"])
        previous_records = int(row["previous_records"])
        delta = int(row["records_delta"])

        if previous_records == 0:
            lines.append(
                f"Para '{behavior}', não houve base suficiente no período anterior para estabelecer comparação direta."
            )
            continue

        if delta > 0:
            if abs(delta) <= max(2, int(previous_records * 0.15)):
                variation = "leve aumento na frequência"
            else:
                variation = "aumento mais perceptível na frequência"
        elif delta < 0:
            if abs(delta) <= max(2, int(previous_records * 0.15)):
                variation = "leve redução na frequência"
            else:
                variation = "redução mais evidente na frequência"
        else:
            variation = "estabilidade na frequência registrada"

        predominance_clause = ""
        if behavior == summary.iloc[0]["behavior"]:
            predominance_clause = ", mantendo, contudo, predominância geral no padrão observado"

        lines.append(
            f"Em relação ao período anterior, as ocorrências classificadas como '{behavior}' apresentaram {variation}{predominance_clause}."
        )

    return lines or ["Não há base comparativa suficiente para o período imediatamente anterior."]


def generate_methodological_note() -> str:
    return (
        "A duração acumulada estimada representa uma aproximação derivada da continuidade dos registros observacionais "
        "persistidos entre os registros de início e término dos episódios. Essa medida pode sofrer impacto da taxa de "
        "amostragem, de perdas momentâneas de detecção, de oclusões e de variações nas condições de captação, não "
        "devendo ser interpretada como medição absoluta e contínua do comportamento."
    )


def build_limitations_text() -> str:
    return (
        "As métricas apresentadas são indicadores observacionais derivados de reconhecimento facial, detecção de pose "
        "e regras geométricas. O sistema é uma prova de conceito e pode sofrer impacto de variações de iluminação, "
        "ângulo da câmera, oclusões, movimentação coletiva, perda de pontos-chave, qualidade do fluxo RTSP e contexto "
        "pedagógico da aula. Os resultados não devem ser interpretados como diagnóstico ou avaliação clínica, mas como "
        "apoio analítico para leitura temporal de recorrências observadas."
    )


def build_report_pdf(report_data: dict) -> bytes:
    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)
    width, height = A4
    margin_x = 42
    top = height - 38
    cursor_y = top
    usable_width = width - (margin_x * 2)
    paragraph_style = ParagraphStyle(
        "ReportBody",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
    )

    def next_page():
        nonlocal cursor_y
        pdf.showPage()
        cursor_y = top

    def ensure_space(required_height: int):
        nonlocal cursor_y
        if cursor_y - required_height < 48:
            next_page()

    def write_line(text: str = "", font_name: str = "Helvetica", font_size: int = 10, color=colors.black):
        nonlocal cursor_y
        ensure_space(16)
        pdf.setFont(font_name, font_size)
        pdf.setFillColor(color)
        pdf.drawString(margin_x, cursor_y, text[:120])
        cursor_y -= 14

    def write_paragraph(text: str):
        nonlocal cursor_y
        safe_text = escape(text).replace("\n", "<br/>")
        paragraph = Paragraph(safe_text, paragraph_style)
        _, paragraph_height = paragraph.wrap(usable_width, top)
        ensure_space(int(paragraph_height) + 4)
        paragraph.drawOn(pdf, margin_x, cursor_y - paragraph_height)
        cursor_y -= paragraph_height + 4

    def write_section(title: str):
        nonlocal cursor_y
        ensure_space(34)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.setFillColor(colors.HexColor("#1F3B63"))
        pdf.drawString(margin_x, cursor_y, title[:120])
        cursor_y -= 8
        pdf.setStrokeColor(colors.HexColor("#C9D2E3"))
        pdf.setLineWidth(0.8)
        pdf.line(margin_x, cursor_y, margin_x + usable_width, cursor_y)
        cursor_y -= 14

    def write_bullets(lines: list[str]):
        for line in lines:
            wrapped = textwrap.wrap(line, width=100) or [""]
            ensure_space(16 * max(len(wrapped), 1))
            first = True
            for wrapped_line in wrapped:
                prefix = u"\u2022 " if first else "  "
                write_line(f"{prefix}{wrapped_line}")
                first = False

    pdf.setTitle("Relatório Observacional de Padrões Comportamentais em Sala de Aula")

    write_line("Relatório Observacional de Padrões Comportamentais em Sala de Aula", "Helvetica-Bold", 15)
    write_line(f"Aluno: {report_data['student']}", "Helvetica", 10)
    write_line(
        f"Período analisado: {format_date_br(report_data['start_date'])} a {format_date_br(report_data['end_date'])} | Recorte: {report_data['period_mode']}",
        "Helvetica",
        10,
    )
    write_line()

    metrics = report_data["headline_metrics"]
    write_section("Resumo Geral")
    write_paragraph(build_observational_summary(report_data))
    write_line()

    write_section("Indicadores Sintéticos")
    write_bullets(
        [
            f"Total de episódios observacionais: {metrics['total_records']}",
            f"Duração acumulada estimada: {format_duration_human(metrics['total_duration_seconds'])}",
            f"Comportamento predominante: {metrics['predominant_behavior']}",
            f"Dias com registros: {metrics['active_days']}",
        ]
    )
    write_line()

    write_section("Leitura Interpretativa do Período")
    write_paragraph(generate_interpretive_summary(report_data))
    write_line()

    write_section("Distribuição Temporal no Período")
    write_paragraph(generate_temporal_distribution_summary(report_data))
    write_line()

    write_section("Consistência Comportamental")
    write_paragraph(generate_consistency_summary(report_data))
    write_line()

    summary = report_data["behavior_summary"]
    write_section("Frequências por Comportamento")
    if summary.empty:
        write_line("Sem dados no período selecionado.")
    else:
        write_bullets(
            [
                (
                    f"Ocorrências classificadas como {row['behavior']}: {int(row['records'])} registro(s), "
                    f"{row['occurrence_percentage']:.2f}% das ocorrências, "
                    f"{row['duration_minutes']:.2f} min acumulados"
                )
                for _, row in summary.iterrows()
            ]
        )
    write_line()

    peak_days = report_data["peak_days"]
    write_section("Dias com Maior Recorrência")
    if peak_days.empty:
        write_line("Sem dados suficientes para destaque temporal.")
    else:
        write_bullets(
            [
                (
                    f"{row['behavior']} em {format_date_br(row['date'])}: "
                    f"{int(row['records'])} registro(s), {row['duration_minutes']:.2f} min"
                )
                for _, row in peak_days.head(6).iterrows()
            ]
        )
    write_line()

    write_section("Pontos de Atenção Observacionais")
    write_bullets(generate_observational_attention_points(report_data))
    write_line()

    write_section("Comparação com o Período Anterior")
    write_bullets(generate_previous_period_comparison(report_data))
    write_line()

    write_section("Nota Metodológica sobre a Duração Estimada")
    write_paragraph(generate_methodological_note())
    write_line()

    write_section("Limitações Metodológicas")
    write_paragraph(build_limitations_text())

    pdf.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
