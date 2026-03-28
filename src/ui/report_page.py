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
    "predominancia": {"accent": "#66BB6A", "icon": "◆", "badge": "Principal"},
    "segunda_recorrencia": {"accent": "#FFB74D", "icon": "◇", "badge": "Complementar"},
    "padrao_geral": {"accent": "#64B5F6", "icon": "≋", "badge": "Consistência"},
    "variacao_principal": {"accent": "#EF5350", "icon": "⇄", "badge": "Comparação"},
    "neutro": {"accent": "#90A4AE", "icon": "•", "badge": "Resumo"},
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


def _build_export_base_name(student_name: str, start_date: date, end_date: date) -> str:
    safe_student = "_".join(str(student_name or "aluno").split()).lower()
    return f"{safe_student}_{start_date.isoformat()}_{end_date.isoformat()}"


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


def _inject_report_top_styles() -> None:
    st.markdown(
        """
        <style>
        .report-top-hero {
            padding: 1.15rem 0 0.35rem 0;
        }
        .report-top-title {
            display: flex;
            align-items: center;
            gap: 0.9rem;
            margin-bottom: 0.4rem;
        }
        .report-top-icon {
            width: 64px;
            height: 64px;
            border-radius: 18px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            background: linear-gradient(180deg, rgba(115,154,255,0.24) 0%, rgba(87,108,196,0.16) 100%);
            border: 1px solid rgba(140,160,255,0.24);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .report-top-heading {
            font-size: 2.3rem;
            line-height: 1.08;
            font-weight: 800;
            color: #F4F6FB;
            margin: 0;
        }
        .report-top-subheading {
            font-size: 1rem;
            line-height: 1.65;
            color: #A7ADBB;
            margin: 0.25rem 0 0.1rem 0;
        }
        .report-top-panel {
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.09);
            background: linear-gradient(180deg, rgba(27,29,38,0.84) 0%, rgba(22,24,33,0.94) 100%);
            padding: 1.1rem 1rem 0.35rem 1rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.18);
            margin: 1rem 0 1rem 0;
        }
        .report-action-shell {
            border-radius: 22px;
            border: 1px solid rgba(123, 144, 255, 0.18);
            background:
                radial-gradient(circle at top right, rgba(86, 112, 255, 0.18), transparent 34%),
                linear-gradient(180deg, rgba(29, 33, 48, 0.94) 0%, rgba(21, 24, 34, 0.98) 100%);
            padding: 1rem 1.1rem 1.05rem 1.1rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.18);
            margin: 0.35rem 0 1rem 0;
        }
        .report-action-eyebrow {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #9BA7C0;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .report-action-title {
            font-size: 1.16rem;
            font-weight: 800;
            color: #F5F7FC;
            margin-bottom: 0.25rem;
        }
        .report-action-subtitle {
            font-size: 0.95rem;
            line-height: 1.6;
            color: #B6BED0;
            margin: 0;
        }
        .report-action-note {
            margin-top: 0.65rem;
            font-size: 0.88rem;
            color: #8F99AE;
        }
        .report-filter-note {
            margin-top: -0.2rem;
            margin-bottom: 0.85rem;
            color: #98A2B8;
            font-size: 0.93rem;
            line-height: 1.55;
        }
        .report-section-kicker {
            font-size: 0.84rem;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #9EA7BC;
            font-weight: 700;
            margin: 0 0 0.85rem 0;
        }
        .report-panel-title {
            font-size: 0.88rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #9EA6B6;
            font-weight: 700;
            margin-bottom: 0.8rem;
        }
        .report-kpi-shell {
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.10);
            background: linear-gradient(180deg, rgba(27,29,38,0.84) 0%, rgba(22,24,33,0.94) 100%);
            box-shadow: 0 16px 36px rgba(0,0,0,0.18);
            padding: 0.8rem 0.8rem 0.6rem 0.8rem;
            margin-bottom: 1rem;
        }
        .report-kpi-card {
            min-height: 290px;
            padding: 1rem 1rem 0.8rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            text-align: center;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        .report-kpi-card.last {
            border-right: none;
        }
        .report-kpi-orb {
            width: 126px;
            height: 126px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 3rem;
            margin-bottom: 1rem;
            box-shadow: inset 0 10px 22px rgba(255,255,255,0.14), 0 12px 30px rgba(0,0,0,0.22);
        }
        .report-kpi-title {
            font-size: 1.02rem;
            line-height: 1.35;
            color: #F3F5F9;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .report-kpi-value {
            font-size: 2rem;
            line-height: 1.1;
            color: #FFFFFF;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .report-kpi-caption {
            font-size: 0.9rem;
            line-height: 1.45;
            color: #A5ACBA;
        }
        .report-kpi-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 150px;
            padding: 0.55rem 1rem;
            border-radius: 999px;
            margin-top: 0.3rem;
            font-size: 0.98rem;
            font-weight: 700;
            color: #F7FBF7;
            background: linear-gradient(90deg, #5C9130 0%, #79A94A 100%);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
        }
        .report-kpi-arc {
            width: 165px;
            height: 84px;
            border-top-left-radius: 170px;
            border-top-right-radius: 170px;
            border: 16px solid rgba(187,212,66,0.55);
            border-bottom: 0;
            position: relative;
            margin: 1rem auto 0.55rem auto;
            box-sizing: border-box;
        }
        .report-kpi-arc::after {
            content: "";
            position: absolute;
            width: 16px;
            height: 4px;
            border-radius: 999px;
            background: #E9E15B;
            right: 10px;
            top: 20px;
            transform: rotate(-30deg);
            box-shadow: 0 0 0 1px rgba(255,255,255,0.06);
        }
        .report-kpi-arc-label {
            font-size: 1rem;
            font-weight: 700;
            color: #F4F6FB;
            margin: -0.15rem 0 0.55rem 0;
        }
        .overview-panel {
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            padding: 0.95rem 1rem;
            margin-bottom: 0.8rem;
        }
        .overview-panel.executive {
            background: linear-gradient(180deg, rgba(56, 78, 112, 0.22) 0%, rgba(39, 49, 74, 0.16) 100%);
        }
        .overview-panel.interpretive {
            background: linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.02) 100%);
        }
        .overview-panel-title {
            font-size: 0.92rem;
            font-weight: 700;
            color: #F2F5FA;
            margin-bottom: 0.45rem;
        }
        .overview-panel-body {
            color: #CDD4DF;
            line-height: 1.68;
            font-size: 0.97rem;
        }
        .overview-panel-body.muted {
            color: #B3BBC8;
            font-size: 0.95rem;
        }
        .attention-list {
            display: grid;
            gap: 0.65rem;
            margin-top: 0.55rem;
        }
        .attention-item {
            display: flex;
            gap: 0.7rem;
            align-items: flex-start;
            padding: 0.72rem 0.82rem;
            border-radius: 14px;
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .attention-marker {
            flex: 0 0 auto;
            width: 0.52rem;
            height: 0.52rem;
            border-radius: 999px;
            background: #FFB74D;
            margin-top: 0.42rem;
            box-shadow: 0 0 0 4px rgba(255,183,77,0.12);
        }
        .attention-text {
            color: #D4DAE4;
            line-height: 1.6;
            font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_report_top_header() -> None:
    st.markdown(
        """
        <div class="report-top-hero">
            <div class="report-top-title">
                <div class="report-top-icon">📑</div>
                <div class="report-top-heading">Relatórios Observacionais</div>
            </div>
            <div class="report-top-subheading">
                Relatório consolidado por aluno, considerando o contexto da sessão monitorada, a disciplina, a turma e o professor.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_report_export_actions(
    report_data: dict,
    display_summary: pd.DataFrame,
    timeline_display: pd.DataFrame,
    selected_student: str,
    start_date: date,
    end_date: date,
) -> None:
    export_base_name = _build_export_base_name(selected_student, start_date, end_date)
    pdf_bytes = build_report_pdf(report_data)
    summary_csv = _to_csv_bytes(display_summary)
    timeline_csv = _to_csv_bytes(timeline_display)

    st.markdown(
        """
        <div class="report-action-shell">
            <div class="report-action-eyebrow">Saída do Relatório</div>
            <div class="report-action-title">Gerar a versão formal do relatório</div>
            <p class="report-action-subtitle">
                Esta tela apresenta uma visão analítica resumida. Para compartilhar, arquivar ou encaminhar o resultado,
                exporte o relatório completo em PDF ou os dados em formato tabular.
            </p>
            <div class="report-action-note">As exportações detalhadas continuam disponíveis na aba “Detalhes e Exportação”.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_col1, action_col2, action_col3 = st.columns([1.15, 1, 1], gap="medium")
    action_col1.download_button(
        "📄 Gerar PDF",
        data=pdf_bytes,
        file_name=f"relatorio_observacional_{export_base_name}.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )
    action_col2.download_button(
        "📊 Exportar resumo",
        data=summary_csv,
        file_name=f"relatorio_resumo_{export_base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    action_col3.download_button(
        "🗂 Exportar episódios",
        data=timeline_csv,
        file_name=f"relatorio_episodios_{export_base_name}.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_kpi_card(title: str, value: str, subtitle: str, orb_style: str, icon: str, is_last: bool = False) -> None:
    extra_class = " last" if is_last else ""
    st.markdown(
        (
            f"<div class='report-kpi-card{extra_class}'>"
            f"<div class='report-kpi-orb' style='{orb_style}'>{html.escape(icon)}</div>"
            f"<div class='report-kpi-title'>{html.escape(title)}</div>"
            f"<div class='report-kpi-value'>{html.escape(value)}</div>"
            f"<div class='report-kpi-caption'>{html.escape(subtitle)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_predominant_behavior_card(behavior: str, share_label: str) -> None:
    behavior_lower = (behavior or "").strip().lower()
    if behavior_lower in {"dormindo", "distraido", "distraído", "agitado"}:
        badge_background = "linear-gradient(90deg, #A93C3C 0%, #D05757 100%)"
        arc_border = "rgba(217,102,102,0.78)"
        arc_pointer = "#FFB0B0"
    elif behavior_lower in {"perguntando", "em pé", "em pe"}:
        badge_background = "linear-gradient(90deg, #83611D 0%, #B58B2A 100%)"
        arc_border = "rgba(213,190,88,0.78)"
        arc_pointer = "#F3E56C"
    else:
        badge_background = "linear-gradient(90deg, #5C9130 0%, #79A94A 100%)"
        arc_border = "rgba(187,212,66,0.55)"
        arc_pointer = "#E9E15B"

    st.markdown(
        (
            "<div class='report-kpi-card'>"
            f"<div class='report-kpi-arc' style='border-color:{arc_border}; border-bottom:0;'>"
            f"<span style='position:absolute; width:16px; height:4px; border-radius:999px; background:{arc_pointer}; "
            "right:10px; top:20px; transform:rotate(-30deg); box-shadow:0 0 0 1px rgba(255,255,255,0.06);'></span>"
            "</div>"
            f"<div class='report-kpi-arc-label'>{html.escape(behavior)}</div>"
            "<div class='report-kpi-title'>Comportamento predominante</div>"
            f"<div class='report-kpi-badge' style='background:{badge_background};'>{html.escape(behavior)}</div>"
            f"<div class='report-kpi-caption' style='margin-top:0.65rem;'>{html.escape(share_label)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _shorten_text(text: str, max_sentences: int = 2) -> str:
    if not text:
        return ""

    normalized = " ".join(str(text).split())
    tokens = normalized.replace("?", ".").replace("!", ".").split(". ")
    sentences: list[str] = []

    for token in tokens:
        candidate = token.strip()
        if not candidate:
            continue
        if candidate[-1] not in ".!?":
            candidate = f"{candidate}."
        sentences.append(candidate)
        if len(sentences) >= max_sentences:
            break

    return " ".join(sentences).strip() if sentences else normalized


def _render_overview_text_panel(title: str, text: str, variant: str = "executive") -> None:
    if not text:
        st.info("Não houve base suficiente para compor este bloco.")
        return

    body_class = "overview-panel-body" if variant == "executive" else "overview-panel-body muted"
    st.markdown(
        (
            f"<div class='overview-panel {variant}'>"
            f"<div class='overview-panel-title'>{html.escape(title)}</div>"
            f"<div class='{body_class}'>{html.escape(text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_attention_points(points: list[str]) -> None:
    if not points:
        st.info("Não houve pontos adicionais de atenção observacional para o período selecionado.")
        return

    items = []
    for point in points:
        concise_point = _shorten_text(point, max_sentences=1)
        items.append(
            "<div class='attention-item'>"
            "<div class='attention-marker'></div>"
            f"<div class='attention-text'>{html.escape(concise_point)}</div>"
            "</div>"
        )

    st.markdown(f"<div class='attention-list'>{''.join(items)}</div>", unsafe_allow_html=True)


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
    _inject_report_top_styles()
    _render_report_top_header()
    top_actions_placeholder = st.empty()

    base_options = get_available_filters(user_context)
    students_df = base_options["students"]
    if students_df.empty:
        st.warning("Não há episódios comportamentais registrados para gerar relatórios.")
        return

    with st.container(border=True):
        st.markdown("<div class='report-panel-title'>Filtros do relatório</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='report-filter-note'>Defina o contexto de análise abaixo. Em seguida, use as ações no topo para gerar a versão formal do relatório ou exportar os dados.</div>",
            unsafe_allow_html=True,
        )

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

        controls_col5, controls_col6 = st.columns([1, 1])
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
    short_summary = _shorten_text(build_observational_summary(report_data), max_sentences=2)
    short_interpretation = _shorten_text(generate_interpretive_summary(report_data), max_sentences=2)
    short_temporal_summary = _shorten_text(_build_timeline_temporal_summary(report_data), max_sentences=2)
    concise_attention_points = attention_points[:3]
    concise_comparison_lines = previous_period_lines[:3]

    with top_actions_placeholder.container():
        _render_report_export_actions(
            report_data=report_data,
            display_summary=display_summary,
            timeline_display=timeline_display,
            selected_student=selected_student,
            start_date=start_date,
            end_date=end_date,
        )

    predominant_share = float(summary_df.iloc[0]["duration_percentage"]) if not summary_df.empty else 0.0
    total_records_subtitle = (
        f"{metrics['total_records']} episódio registrado"
        if metrics["total_records"] == 1
        else f"{metrics['total_records']} episódios registrados"
    )
    active_days_subtitle = (
        f"{metrics['active_days']} dia com registros"
        if metrics["active_days"] == 1
        else f"{metrics['active_days']} dias com registros"
    )

    with st.container(border=True):
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            _render_kpi_card(
                "Episódios",
                str(metrics["total_records"]),
                total_records_subtitle,
                "background: radial-gradient(circle at 32% 28%, #5F90D2 0%, #426CA8 58%, #35527F 100%);",
                "📄",
            )
        with metric_col2:
            _render_kpi_card(
                "Dias com registros",
                str(metrics["active_days"]),
                active_days_subtitle,
                "background: radial-gradient(circle at 32% 28%, #88B85A 0%, #68983F 58%, #4E7730 100%);",
                "🗓",
            )
        with metric_col3:
            _render_predominant_behavior_card(
                str(metrics["predominant_behavior"]),
                f"({predominant_share:.0f}% da duração total)",
            )
        with metric_col4:
            _render_kpi_card(
                "Duração acumulada",
                format_duration_human(metrics["total_duration_seconds"]),
                "Tempo total registrado",
                "background: radial-gradient(circle at 32% 28%, #F1B24D 0%, #E8962F 58%, #C9771B 100%);",
                "🕒",
                is_last=True,
            )
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

    hourly_summary_display = _build_hourly_summary_display(timeline_df)
    tabs = st.tabs(
        ["Visão Geral", "Frequência e Duração", "Distribuição Temporal", "Comparações", "Detalhes e Exportação"]
    )

    with tabs[0]:
        st.subheader("Síntese Analítica")
        overview_col1, overview_col2 = st.columns([1.35, 1], gap="large")
        with overview_col1:
            _render_overview_text_panel("Resumo Executivo", short_summary, variant="executive")
            _render_overview_text_panel("Leitura Interpretativa", short_interpretation, variant="interpretive")
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
        _render_attention_points(concise_attention_points)

    with tabs[1]:
        st.subheader("Frequência e Duração")
        st.caption("Distribuição dos comportamentos observados e apoio visual de duração estimada.")
        chart_col1, chart_col2 = st.columns([1.2, 1], gap="large")
        with chart_col1:
            st.plotly_chart(fig_occurrence, width="stretch")
        with chart_col2:
            st.plotly_chart(fig_duration, width="stretch")
        with st.expander("Ver tabela consolidada de frequências", expanded=False):
            st.dataframe(display_summary, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Distribuição Temporal")
        st.caption("Linha do tempo dos episódios observados ao longo do período monitorado.")
        st.plotly_chart(fig_timeline, width="stretch")
        st.info(short_temporal_summary or "Não houve base suficiente para sintetizar a leitura temporal.")

    with tabs[3]:
        st.subheader("Comparações")
        st.caption("Síntese curta da comparação com o período anterior e, quando disponível, com recortes internos.")
        st.markdown("##### Síntese Comparativa")
        if concise_comparison_lines:
            for line in concise_comparison_lines:
                st.markdown(f"- {line}")
        else:
            st.info("Não há base comparativa suficiente para uma síntese resumida.")

        if fig_period is not None:
            st.markdown("##### Comparação entre Recortes do Período")
            st.plotly_chart(fig_period, width="stretch")

        with st.expander("Ver comparação detalhada", expanded=False):
            if comparison_display.empty:
                st.info("Não há base comparativa suficiente.")
            else:
                st.dataframe(comparison_display, width="stretch", hide_index=True)

    with tabs[4]:
        st.subheader("Detalhes e Exportação")
        st.caption("Área complementar com tabelas extensas, síntese completa e exportações adicionais.")
        detail_col1, detail_col2 = st.columns([1, 1], gap="large")
        with detail_col1:
            with st.expander("Tabela consolidada de frequências", expanded=False):
                st.dataframe(display_summary, width="stretch", hide_index=True)
            with st.expander("Resumo complementar por faixa horária", expanded=False):
                if hourly_summary_display.empty:
                    st.info("Não houve base suficiente para montar o resumo complementar por faixas horárias.")
                else:
                    st.dataframe(hourly_summary_display, width="stretch", hide_index=True)
            with st.expander("Comparação detalhada com o período anterior", expanded=False):
                if comparison_display.empty:
                    st.info("Não há base comparativa suficiente.")
                else:
                    st.dataframe(comparison_display, width="stretch", hide_index=True)
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
            export_base_name = _build_export_base_name(selected_student, start_date, end_date)
            st.markdown("##### Exportação")
            export_col1, export_col2, export_col3 = st.columns(3)
            export_col1.download_button(
                "Baixar CSV do resumo",
                data=_to_csv_bytes(display_summary),
                file_name=f"relatorio_resumo_{export_base_name}.csv",
                mime="text/csv",
                width="stretch",
            )
            export_col2.download_button(
                "Baixar CSV dos episódios",
                data=_to_csv_bytes(timeline_display),
                file_name=f"relatorio_episodios_{export_base_name}.csv",
                mime="text/csv",
                width="stretch",
            )
            export_col3.download_button(
                "Baixar PDF",
                data=build_report_pdf(report_data),
                file_name=f"relatorio_observacional_{export_base_name}.pdf",
                mime="application/pdf",
                width="stretch",
            )

            with st.expander("Síntese analítica completa", expanded=False):
                st.markdown("##### Resumo Geral")
                _render_justified_text(build_observational_summary(report_data))
                st.markdown("##### Leitura Interpretativa")
                _render_justified_text(generate_interpretive_summary(report_data))
            with st.expander("Limitações e nota metodológica", expanded=False):
                st.markdown("##### Limitações Metodológicas")
                _render_justified_text(build_limitations_text())
                st.markdown("##### Nota Metodológica sobre a Duração Estimada")
                _render_justified_text(generate_methodological_note())
