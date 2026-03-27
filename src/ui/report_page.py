from __future__ import annotations

import html
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from services.report_service import generate_report_data, get_available_filters, get_available_students
from utils.report_formatters import (
    build_limitations_text,
    build_observational_summary,
    build_report_pdf,
    format_date_br,
    format_duration_human,
    generate_consistency_summary,
    generate_interpretive_summary,
    generate_methodological_note,
    generate_observational_attention_points,
    generate_previous_period_comparison,
)


BEHAVIOR_COLORS = {
    "Atento": "#43A047",
    "Distraido": "#C0392B",
    "Distraído": "#C0392B",
    "Perguntando": "#1E88E5",
    "Escrevendo": "#16A085",
    "Dormindo": "#8E44AD",
    "Agitado": "#FB8C00",
    "Em Pé": "#7F8C8D",
}

HIGHLIGHT_CARD_STYLES = {
    "predominancia": {"accent": "#66BB6A", "icon": "●", "badge": "Principal"},
    "segunda_recorrencia": {"accent": "#FFB74D", "icon": "●", "badge": "Complementar"},
    "padrao_geral": {"accent": "#64B5F6", "icon": "●", "badge": "Consistência"},
    "variacao_principal": {"accent": "#EF5350", "icon": "●", "badge": "Comparação"},
    "neutro": {"accent": "#90A4AE", "icon": "●", "badge": "Resumo"},
}

DEFAULT_CLASS_START = pd.Timestamp("2000-01-01 08:00:00")
DEFAULT_CLASS_END = pd.Timestamp("2000-01-01 11:40:00")


def _default_range(period_mode: str):
    today = date.today()
    if period_mode == "Diário":
        return today, today
    if period_mode == "Semanal":
        return today - timedelta(days=6), today
    return today - timedelta(days=29), today


def _get_student_last_date(students_df: pd.DataFrame, student_name: str) -> date | None:
    if students_df.empty or not student_name:
        return None
    row = students_df[students_df["student"] == student_name]
    if row.empty:
        return None
    last_date = pd.to_datetime(row.iloc[0]["last_date"], errors="coerce")
    if pd.isna(last_date):
        return None
    return last_date.date()


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8-sig")


def _render_justified_text(text: str) -> None:
    if not text:
        return
    safe_text = html.escape(text)
    safe_text = safe_text.replace("\n", "<br>")
    st.markdown(
        (
            "<div style='text-align: justify; line-height: 1.7; margin-bottom: 0.5rem;'>"
            f"{safe_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_summary_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df.rename(
        columns={
            "behavior": "Comportamento",
            "records": "Registros",
            "occurrence_percentage": "Ocorrência relativa (%)",
            "duration_minutes": "Duração estimada (min)",
            "duration_percentage": "Participação na duração (%)",
        }
    )[
        [
            "Comportamento",
            "Registros",
            "Ocorrência relativa (%)",
            "Duração estimada (min)",
            "Participação na duração (%)",
        ]
    ]


def _build_timeline_display(timeline_df: pd.DataFrame) -> pd.DataFrame:
    timeline_display = timeline_df.copy()
    timeline_display["date"] = pd.to_datetime(timeline_display["date"]).dt.strftime("%d-%m-%Y")
    timeline_display["start_time"] = pd.to_datetime(timeline_display["start_time"]).dt.strftime("%d-%m-%Y %H:%M:%S")
    timeline_display["end_time"] = pd.to_datetime(timeline_display["end_time"]).dt.strftime("%d-%m-%Y %H:%M:%S")
    timeline_display = timeline_display.rename(
        columns={
            "date": "Data",
            "start_time": "Início",
            "end_time": "Fim",
            "behavior": "Comportamento",
            "duration_minutes": "Duração estimada (min)",
            "discipline": "Disciplina",
            "teacher": "Professor",
            "source": "Origem",
        }
    )
    return timeline_display[
        ["Data", "Início", "Fim", "Comportamento", "Duração estimada (min)", "Disciplina", "Professor", "Origem"]
    ]


def _build_consistency_display(consistency_df: pd.DataFrame) -> pd.DataFrame:
    if consistency_df.empty:
        return consistency_df
    return consistency_df.rename(
        columns={
            "behavior": "Comportamento",
            "days_with_occurrence": "Dias com ocorrência",
            "active_day_percentage": "Cobertura dos Dias (%)",
            "total_records": "Registros totais",
            "max_daily_records": "Pico diário de registros",
            "max_daily_share_percentage": "Concentração no pico diário (%)",
            "consistency_label": "Classificação de consistência",
        }
    )


def _build_comparison_display(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df.empty:
        return comparison_df
    return comparison_df.rename(
        columns={
            "behavior": "Comportamento",
            "current_records": "Registros atuais",
            "previous_records": "Registros anteriores",
            "records_delta": "Variação de registros",
            "current_duration_minutes": "Duração atual (min)",
            "previous_duration_minutes": "Duração anterior (min)",
            "duration_delta_minutes": "Variação de duração (min)",
        }
    )


def _build_peak_days_display(peak_days_df: pd.DataFrame) -> pd.DataFrame:
    if peak_days_df.empty:
        return peak_days_df
    peak_days = peak_days_df.rename(
        columns={
            "behavior": "Comportamento",
            "date": "Data",
            "records": "Registros",
            "duration_minutes": "Duração estimada (min)",
        }
    ).copy()
    peak_days["Data"] = pd.to_datetime(peak_days["Data"]).dt.strftime("%d-%m-%Y")
    return peak_days


def _build_hourly_summary_display(timeline_df: pd.DataFrame) -> pd.DataFrame:
    if timeline_df.empty:
        return pd.DataFrame()

    hourly = timeline_df.copy()
    hourly["start_time"] = pd.to_datetime(hourly["start_time"])
    hourly["hour_bucket"] = hourly["start_time"].dt.floor("h")
    hourly["Faixa horária"] = (
        hourly["hour_bucket"].dt.strftime("%H:%M")
        + "-"
        + (hourly["hour_bucket"] + pd.Timedelta(minutes=59)).dt.strftime("%H:%M")
    )
    hourly_summary = (
        hourly.groupby(["Faixa horária", "behavior"], as_index=False)
        .size()
        .rename(columns={"behavior": "Comportamento", "size": "Registros"})
        .pivot(index="Faixa horária", columns="Comportamento", values="Registros")
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    return hourly_summary


def _build_timeline_temporal_summary(report_data: dict) -> str:
    timeline_df = report_data["timeline"]
    summary_df = report_data["behavior_summary"]

    if timeline_df.empty or summary_df.empty:
        return (
            "Não houve base temporal suficiente para sintetizar a sequência cronológica dos episódios observados."
        )

    timeline = timeline_df.copy()
    timeline["start_time"] = pd.to_datetime(timeline["start_time"])
    timeline["hour_bucket"] = timeline["start_time"].dt.floor("h")
    hourly_totals = (
        timeline.groupby("hour_bucket", as_index=False)
        .size()
        .rename(columns={"size": "records"})
        .sort_values(["records", "hour_bucket"], ascending=[False, True])
    )
    total_records = max(int(summary_df["records"].sum()), 1)
    top_behavior = summary_df.iloc[0]
    second_behavior = summary_df.iloc[1] if len(summary_df) > 1 else None
    active_days = int(timeline["date"].nunique())

    top_bucket = hourly_totals.iloc[0]
    top_bucket_label = (
        f"{top_bucket['hour_bucket'].strftime('%H:%M')}-"
        f"{(top_bucket['hour_bucket'] + pd.Timedelta(minutes=59)).strftime('%H:%M')}"
    )
    top_bucket_share = float(top_bucket["records"]) / total_records * 100.0

    if active_days == 1:
        opening = (
            f"A leitura temporal indica predominância de registros classificados como '{top_behavior['behavior']}' "
            "ao longo de grande parte da aula monitorada."
        )
    else:
        opening = (
            f"A leitura temporal indica predominância de registros classificados como '{top_behavior['behavior']}' "
            f"ao longo dos {active_days} dias contemplados no período analisado."
        )

    if second_behavior is not None and float(second_behavior["occurrence_percentage"]) >= 10.0:
        complementary = (
            f" Também foram observadas ocorrências de episódios classificados como '{second_behavior['behavior']}' "
            "em momentos específicos do período monitorado."
        )
    else:
        complementary = ""

    if top_bucket_share >= 35.0:
        distribution_text = (
            f" Houve maior concentração relativa de registros na faixa horária de {top_bucket_label}, "
            "o que sugere maior densidade observacional nesse trecho do período."
        )
    else:
        distribution_text = (
            " Os registros permaneceram distribuídos ao longo de diferentes momentos do período observado, "
            "sem concentração temporal muito acentuada em uma única faixa horária."
        )

    return (
        f"{opening}{complementary}{distribution_text} "
        "Essa leitura deve ser interpretada de forma cautelosa e sempre articulada ao contexto pedagógico, "
        "ao tipo de atividade em curso e às condições de observação."
    )


def _format_duration_br(total_seconds: float) -> str:
    total_seconds = int(round(float(total_seconds or 0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours} h {minutes} min {seconds} s"
    if minutes:
        return f"{minutes} min {seconds} s"
    return f"{seconds} s"


def _get_timeline_dtick_ms(span_minutes: float) -> int:
    if span_minutes <= 30:
        return 5 * 60 * 1000
    if span_minutes <= 90:
        return 10 * 60 * 1000
    if span_minutes <= 180:
        return 15 * 60 * 1000
    if span_minutes <= 360:
        return 30 * 60 * 1000
    return 60 * 60 * 1000


def _build_timeline_figure(timeline_df: pd.DataFrame) -> px.timeline:
    timeline_plot_df = timeline_df.copy()
    timeline_plot_df["duration_label"] = timeline_plot_df["duration_seconds"].apply(_format_duration_br)
    timeline_plot_df["start_label"] = pd.to_datetime(timeline_plot_df["start_time"]).dt.strftime("%H:%M:%S")
    timeline_plot_df["end_label"] = pd.to_datetime(timeline_plot_df["end_time"]).dt.strftime("%H:%M:%S")
    timeline_plot_df["date_label_br"] = pd.to_datetime(timeline_plot_df["date"]).dt.strftime("%d/%m/%Y")

    fig_timeline = px.timeline(
        timeline_plot_df,
        x_start="timeline_start",
        x_end="timeline_end",
        y="date_label",
        color="behavior",
        title="Linha do tempo da aula",
        labels={
            "behavior": "Comportamento",
            "date_label": "Data",
        },
        color_discrete_map=BEHAVIOR_COLORS,
        custom_data=["behavior", "date_label_br", "start_label", "end_label", "duration_label"],
    )
    fig_timeline.update_yaxes(
        autorange="reversed",
        title_text="Data",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.06)",
        tickfont=dict(size=12),
    )

    timeline_start = timeline_plot_df["timeline_start"].min()
    timeline_end = timeline_plot_df["timeline_end"].max()
    visual_start = min(DEFAULT_CLASS_START, timeline_start) if pd.notna(timeline_start) else DEFAULT_CLASS_START
    visual_end = max(DEFAULT_CLASS_END, timeline_end) if pd.notna(timeline_end) else DEFAULT_CLASS_END
    span_minutes = max(
        1.0,
        float((visual_end - visual_start).total_seconds()) / 60.0 if pd.notna(visual_start) and pd.notna(visual_end) else 1.0,
    )
    margin_minutes = 5

    if pd.notna(visual_start) and pd.notna(visual_end):
        fig_timeline.update_xaxes(
            range=[
                visual_start - pd.Timedelta(minutes=margin_minutes),
                visual_end + pd.Timedelta(minutes=margin_minutes),
            ]
        )

    fig_timeline.update_xaxes(
        tickformat="%H:%M",
        dtick=_get_timeline_dtick_ms(span_minutes),
        title_text="Horário real da aula",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        tickfont=dict(size=12),
    )
    fig_timeline.update_traces(
        width=0.42,
        opacity=0.88,
        marker_line_color="rgba(12,16,24,0.9)",
        marker_line_width=1.2,
        hovertemplate=(
            "<b>Comportamento:</b> %{customdata[0]}<br>"
            "<b>Data:</b> %{customdata[1]}<br>"
            "<b>Início:</b> %{customdata[2]}<br>"
            "<b>Fim:</b> %{customdata[3]}<br>"
            "<b>Duração:</b> %{customdata[4]}"
            "<extra></extra>"
        ),
    )
    fig_timeline.update_layout(
        height=max(420, 130 + (timeline_plot_df["date_label"].nunique() * 72)),
        margin=dict(l=10, r=10, t=68, b=22),
        legend_title_text="Comportamento",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        bargap=0.48,
        hoverlabel=dict(
            bgcolor="rgba(20,24,31,0.97)",
            bordercolor="rgba(255,255,255,0.08)",
            font=dict(size=12),
        ),
        title=dict(font=dict(size=18)),
    )
    return fig_timeline


def _build_status_badge(value: str) -> tuple[str | None, str]:
    value_lower = value.lower()
    if "regular" in value_lower:
        return "#64B5F6", "Regular"
    if "episódica" in value_lower or "episodica" in value_lower:
        return "#FFB74D", "Episódica"
    if "intermediária" in value_lower or "intermediaria" in value_lower:
        return "#90A4AE", "Intermediária"
    if "aumento" in value_lower:
        return "#FFB74D", "Aumento"
    if "redução" in value_lower or "reducao" in value_lower:
        return "#EF5350", "Redução"
    if "estabilidade" in value_lower:
        return "#64B5F6", "Estabilidade"
    return None, ""


def _render_highlight_card(kind: str, title: str, value: str, supporting_text: str) -> None:
    style = HIGHLIGHT_CARD_STYLES.get(kind, HIGHLIGHT_CARD_STYLES["neutro"])
    accent = style["accent"]
    icon = style["icon"]
    badge_label = style["badge"]
    status_badge_color, status_badge_text = _build_status_badge(value)
    badge_markup = (
        "<div style='display: flex; gap: 0.45rem; align-items: center; flex-wrap: wrap; margin-bottom: 0.55rem;'>"
        f"<span style='display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.16rem 0.55rem; "
        f"border-radius: 999px; border: 1px solid {accent}55; background: {accent}14; color: {accent}; "
        f"font-size: 0.74rem; font-weight: 600; letter-spacing: 0.01em;'>{html.escape(icon)} {html.escape(badge_label)}</span>"
    )
    if status_badge_text:
        badge_markup += (
            f"<span style='display: inline-flex; align-items: center; padding: 0.16rem 0.55rem; border-radius: 999px; "
            f"border: 1px solid {status_badge_color}55; background: {status_badge_color}14; color: {status_badge_color}; "
            f"font-size: 0.74rem; font-weight: 600; letter-spacing: 0.01em;'>{html.escape(status_badge_text)}</span>"
        )
    badge_markup += "</div>"

    st.markdown(
        (
            "<div style='padding: 1rem 1rem 0.9rem 1rem; border: 1px solid rgba(255,255,255,0.08); "
            f"border-left: 4px solid {accent}; border-radius: 0.95rem; background: linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.015) 100%); "
            "box-shadow: inset 0 1px 0 rgba(255,255,255,0.03); min-height: 158px;'>"
            f"<div style='font-size: 0.82rem; color: {accent}; margin-bottom: 0.42rem; font-weight: 600; letter-spacing: 0.01em;'>{html.escape(title)}</div>"
            f"<div style='font-size: 1.15rem; font-weight: 600; color: #F5F7FA; margin-bottom: 0.55rem;'>{html.escape(value)}</div>"
            f"{badge_markup}"
            f"<div style='font-size: 0.92rem; line-height: 1.55; color: #D8DEE7; text-align: justify;'>{html.escape(supporting_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_quick_insights(report_data: dict) -> dict[str, str]:
    summary_df = report_data["behavior_summary"]
    consistency_df = report_data["behavior_consistency"]
    comparison_df = report_data["comparison"]
    attention_points = generate_observational_attention_points(report_data)

    predominant_behavior = "Sem dados"
    second_behavior = "Sem destaque secundário"
    consistency_label = "Sem base suficiente"
    primary_attention = attention_points[0] if attention_points else "Sem pontos adicionais de atenção."
    primary_variation = "Sem base comparativa suficiente."

    if not summary_df.empty:
        predominant_behavior = str(summary_df.iloc[0]["behavior"])
        if len(summary_df) > 1:
            second_row = summary_df.iloc[1]
            second_behavior = (
                f"{second_row['behavior']} ({float(second_row['occurrence_percentage']):.2f}% dos registros)"
            )

    if not consistency_df.empty and not summary_df.empty:
        consistency_match = consistency_df[consistency_df["behavior"] == predominant_behavior]
        if not consistency_match.empty:
            consistency_label = str(consistency_match.iloc[0]["consistency_label"])

    comparison_lines = generate_previous_period_comparison(report_data)
    if comparison_lines:
        primary_variation = comparison_lines[0]

    return {
        "predominancia": predominant_behavior,
        "segunda_recorrencia": second_behavior,
        "padrao_geral": consistency_label,
        "ponto_atencao": primary_attention,
        "variacao_principal": primary_variation,
    }


def render_report_page(user_context: dict):
    st.title("📑 Relatórios Observacionais")
    st.caption(
        "Relatório consolidado por aluno, considerando o contexto da sessão monitorada, a disciplina, a turma e o professor."
    )

    base_options = get_available_filters(user_context)
    students_df = base_options["students"]
    if students_df.empty:
        st.warning("Não há episódios comportamentais registrados para gerar relatórios.")
        return

    controls_col1, controls_col2, controls_col3, controls_col4 = st.columns([2, 2, 2, 2])
    with controls_col1:
        selected_teacher_id = None
        if user_context["role"] == "admin" and not base_options["teachers"].empty:
            teacher_map = {int(row["id"]): row["nome"] for _, row in base_options["teachers"].iterrows()}
            teacher_choice = st.selectbox("Professor", ["Todos"] + list(teacher_map.values()), index=0)
            if teacher_choice != "Todos":
                selected_teacher_id = next(key for key, value in teacher_map.items() if value == teacher_choice)
        else:
            st.text_input("Professor", value=user_context["name"], disabled=True)
    with controls_col2:
        options_after_teacher = get_available_filters(
            user_context,
            filters={"teacher_id": selected_teacher_id} if selected_teacher_id else None,
        )
        subject_map = {int(row["id"]): row["nome"] for _, row in options_after_teacher["subjects"].iterrows()}
        subject_choice = st.selectbox("Disciplina", ["Todas"] + list(subject_map.values()), index=0)
        selected_subject_id = None
        if subject_choice != "Todas":
            selected_subject_id = next(key for key, value in subject_map.items() if value == subject_choice)
    with controls_col3:
        options_after_subject = get_available_filters(
            user_context,
            filters={
                "teacher_id": selected_teacher_id,
                "subject_id": selected_subject_id,
            },
        )
        class_map = {
            int(row["id"]): row["nome"] if not row["identificador"] else f"{row['nome']} - {row['identificador']}"
            for _, row in options_after_subject["classes"].iterrows()
        }
        class_choice = st.selectbox("Turma", ["Todas"] + list(class_map.values()), index=0)
        selected_class_id = None
        if class_choice != "Todas":
            selected_class_id = next(key for key, value in class_map.items() if value == class_choice)
    with controls_col4:
        filtered_students_df = get_available_students(
            user_context,
            filters={
                "teacher_id": selected_teacher_id,
                "subject_id": selected_subject_id,
                "class_id": selected_class_id,
            },
        )
        student_options = filtered_students_df["student"].tolist()
        if not student_options:
            st.warning("Não há alunos com episódios para os filtros selecionados.")
            return
        selected_student = st.selectbox("Aluno", student_options, index=0)

    controls_col5, controls_col6 = st.columns([1, 2])
    with controls_col5:
        period_mode = st.selectbox("Período", ["Diário", "Semanal", "Mensal"], index=0)
    with controls_col6:
        start_default, end_default = _default_range(period_mode)
        student_last_date = _get_student_last_date(filtered_students_df, selected_student)
        if period_mode == "Diário":
            selected_date = st.date_input("Data de referência", value=student_last_date or end_default)
            start_date = selected_date
            end_date = selected_date
        else:
            selected_range = st.date_input(
                "Intervalo de datas",
                value=(
                    start_default,
                    student_last_date or end_default,
                ),
            )

            if not isinstance(selected_range, (list, tuple)) or len(selected_range) != 2:
                st.info("Selecione uma data inicial e uma data final para gerar o relatório.")
                return

            start_date, end_date = selected_range

    if start_date > end_date:
        st.error("A data inicial não pode ser maior que a data final.")
        return

    report_data = generate_report_data(
        user_context=user_context,
        filters={
            "teacher_id": selected_teacher_id,
            "subject_id": selected_subject_id,
            "class_id": selected_class_id,
            "student_name": selected_student,
            "start_date": start_date,
            "end_date": end_date,
        },
        period_mode=period_mode,
    )

    episodes = report_data["episodes"]
    if episodes.empty:
        if period_mode == "Diário":
            last_available_date = _get_student_last_date(filtered_students_df, selected_student)
            if last_available_date and last_available_date != start_date:
                st.warning(
                    "Não há registros para o aluno na data selecionada. "
                    f"A última data com episódios para este aluno é {format_date_br(last_available_date)}."
                )
            else:
                st.warning("Não há registros para o aluno e o período selecionados.")
        else:
            st.warning("Não há registros para o aluno e o período selecionados.")
        return

    summary_df = report_data["behavior_summary"]
    period_df = report_data["period_distribution"]
    timeline_df = report_data["timeline"]
    comparison_df = report_data["comparison"]
    consistency_df = report_data["behavior_consistency"]
    metrics = report_data["headline_metrics"]
    attention_points = generate_observational_attention_points(report_data)
    previous_period_lines = generate_previous_period_comparison(report_data)
    display_summary = _build_summary_display(summary_df)
    timeline_display = _build_timeline_display(timeline_df)
    consistency_display = _build_consistency_display(consistency_df)
    comparison_display = _build_comparison_display(comparison_df)
    peak_days_display = _build_peak_days_display(report_data["peak_days"])
    quick_insights = _build_quick_insights(report_data)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Episódios", metrics["total_records"])
    metric_col2.metric("Dias com registros", metrics["active_days"])
    metric_col3.metric("Comportamento predominante", metrics["predominant_behavior"])
    metric_col4.metric("Duração acumulada", format_duration_human(metrics["total_duration_seconds"]))
    st.divider()

    fig_occurrence = px.pie(
        summary_df,
        values="records",
        names="behavior",
        hole=0.45,
        title="Ocorrência relativa por comportamento",
        color="behavior",
        color_discrete_map=BEHAVIOR_COLORS,
        labels={"behavior": "Comportamento", "records": "Registros"},
    )
    fig_occurrence.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Comportamento")

    fig_duration = px.bar(
        summary_df,
        x="behavior",
        y="duration_minutes",
        color="behavior",
        title="Duração estimada por comportamento",
        labels={"behavior": "Comportamento", "duration_minutes": "Minutos"},
        color_discrete_map=BEHAVIOR_COLORS,
        text="duration_minutes",
    )
    fig_duration.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_duration.update_layout(margin=dict(l=10, r=10, t=50, b=10), showlegend=False)

    fig_timeline = _build_timeline_figure(timeline_df)

    fig_period = None
    if period_df["period_label"].nunique() > 1:
        fig_period = px.bar(
            period_df,
            x="period_label",
            y="records",
            color="behavior",
            barmode="group",
            title=f"Comparação {period_mode.lower()} dentro do intervalo",
            labels={"period_label": "Recorte", "records": "Registros", "behavior": "Comportamento"},
            color_discrete_map=BEHAVIOR_COLORS,
        )
        fig_period.update_layout(margin=dict(l=10, r=10, t=50, b=10), legend_title_text="Comportamento")

    temporal_summary = _build_timeline_temporal_summary(report_data)
    hourly_summary_display = _build_hourly_summary_display(timeline_df)
    tabs = st.tabs(
        ["Visão Geral", "Frequência e Duração", "Distribuição Temporal", "Comparações", "Detalhes e Exportação"]
    )

    with tabs[0]:
        st.subheader("Síntese Analítica")
        overview_col1, overview_col2 = st.columns([1.35, 1], gap="large")
        with overview_col1:
            st.markdown("##### Resumo Geral")
            _render_justified_text(build_observational_summary(report_data))
            st.markdown("##### Leitura Interpretativa")
            _render_justified_text(generate_interpretive_summary(report_data))
        with overview_col2:
            st.markdown("##### Destaques Rápidos")
            card_col1, card_col2 = st.columns(2)
            with card_col1:
                _render_highlight_card(
                    "predominancia",
                    "Predominância",
                    quick_insights["predominancia"],
                    "Comportamento com maior recorrência no período selecionado.",
                )
            with card_col2:
                _render_highlight_card(
                    "segunda_recorrencia",
                    "Segunda maior recorrência",
                    quick_insights["segunda_recorrencia"],
                    "Ajuda a identificar o comportamento complementar mais frequente.",
                )
            card_col3, card_col4 = st.columns(2)
            with card_col3:
                _render_highlight_card(
                    "padrao_geral",
                    "Padrão geral",
                    quick_insights["padrao_geral"],
                    "Classificação observacional associada à regularidade do comportamento predominante.",
                )
            with card_col4:
                _render_highlight_card(
                    "variacao_principal",
                    "Variação principal",
                    "Período anterior",
                    quick_insights["variacao_principal"],
                )

        st.markdown("##### Pontos de Atenção")
        if attention_points:
            for point in attention_points:
                st.markdown(f"- {point}")
        else:
            st.info("Não houve pontos adicionais de atenção observacional para o período selecionado.")

    with tabs[1]:
        st.subheader("Frequência e Duração")
        st.caption(
            "Esta seção concentra a leitura quantitativa por comportamento, reunindo frequência observada, participação relativa e duração estimada acumulada."
        )
        table_col, chart_col = st.columns([1.15, 1], gap="large")
        with table_col:
            st.markdown("##### Tabela de Frequências")
            st.dataframe(display_summary, width="stretch", hide_index=True)
        with chart_col:
            st.markdown("##### Leitura Visual")
            chart_inner_col1, chart_inner_col2 = st.columns(2)
            with chart_inner_col1:
                st.plotly_chart(fig_occurrence, width="stretch")
            with chart_inner_col2:
                st.plotly_chart(fig_duration, width="stretch")

    with tabs[2]:
        st.subheader("Distribuição Temporal")
        st.caption(
            "A análise temporal abaixo apresenta a sequência cronológica dos comportamentos observados ao longo do período monitorado, preservando a ordem e a duração relativa dos episódios registrados."
        )
        st.caption(
            "Cada bloco representa um episódio contínuo classificado pelo sistema ao longo do horário monitorado."
        )
        st.plotly_chart(fig_timeline, width="stretch")
        _render_justified_text(temporal_summary)
        st.markdown("##### Resumo complementar por faixas horárias reais")
        st.caption(
            "O quadro abaixo agrega os registros por faixa horária real, funcionando apenas como apoio à leitura cronológica principal."
        )
        if hourly_summary_display.empty:
            st.info("Não houve base suficiente para montar o resumo complementar por faixas horárias.")
        else:
            st.dataframe(hourly_summary_display, width="stretch", hide_index=True)

    with tabs[3]:
        st.subheader("Comparações")
        st.caption(
            "Esta seção concentra as variações entre o período analisado e o período imediatamente anterior, além de comparações internas quando o intervalo permite múltiplos recortes."
        )
        st.markdown("##### Síntese Comparativa")
        for line in previous_period_lines:
            st.markdown(f"- {line}")

        if fig_period is not None:
            st.markdown("##### Comparação entre Recortes do Período")
            st.plotly_chart(fig_period, width="stretch")

        st.markdown("##### Comparação com o Período Imediatamente Anterior")
        if comparison_display.empty:
            st.info("Não há base comparativa suficiente.")
        else:
            st.dataframe(comparison_display, width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader("Detalhes e Exportação")
        detail_col1, detail_col2 = st.columns([1, 1], gap="large")
        with detail_col1:
            with st.expander("Resumo cronológico dos episódios", expanded=False):
                st.dataframe(timeline_display, width="stretch", hide_index=True)
            with st.expander("Consistência comportamental detalhada", expanded=False):
                if consistency_display.empty:
                    st.info("Não houve base suficiente para detalhar a consistência comportamental.")
                else:
                    st.dataframe(consistency_display, width="stretch", hide_index=True)
            with st.expander("Dias com maior recorrência por comportamento", expanded=False):
                if peak_days_display.empty:
                    st.info("Não houve dados suficientes para destacar os dias de maior recorrência.")
                else:
                    st.dataframe(peak_days_display, width="stretch", hide_index=True)
        with detail_col2:
            st.markdown("##### Exportação")
            export_col1, export_col2, export_col3 = st.columns(3)
            export_col1.download_button(
                "Baixar CSV do resumo",
                data=_to_csv_bytes(display_summary),
                file_name=f"relatorio_resumo_{selected_student}_{format_date_br(start_date)}_{format_date_br(end_date)}.csv",
                mime="text/csv",
                width="stretch",
            )
            export_col2.download_button(
                "Baixar CSV dos episódios",
                data=_to_csv_bytes(timeline_display),
                file_name=f"relatorio_episodios_{selected_student}_{format_date_br(start_date)}_{format_date_br(end_date)}.csv",
                mime="text/csv",
                width="stretch",
            )
            export_col3.download_button(
                "Baixar PDF",
                data=build_report_pdf(report_data),
                file_name=f"relatorio_observacional_{selected_student}_{format_date_br(start_date)}_{format_date_br(end_date)}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            st.markdown("##### Limitações Metodológicas")
            _render_justified_text(build_limitations_text())
            st.markdown("##### Nota Metodológica sobre a Duração Estimada")
            _render_justified_text(generate_methodological_note())
