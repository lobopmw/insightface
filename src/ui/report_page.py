from __future__ import annotations

import html
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    "atencao_pedagogica": {"accent": "#F6C453", "icon": "▲", "badge": "Observação"},
    "neutro": {"accent": "#90A4AE", "icon": "•", "badge": "Resumo"},
}

DEFAULT_CLASS_START = pd.Timestamp("2000-01-01 08:00:00")
DEFAULT_CLASS_END = pd.Timestamp("2000-01-01 11:40:00")

BEHAVIOR_SIGNAL_PRIORITY_COLORS = {
    "Baixa prioridade": "#4CAF50",
    "Atenção moderada": "#FBC02D",
    "Alta prioridade": "#E53935",
}

MANAGEMENT_CATEGORY_COLORS = {
    "Engajamento na atividade": "#43A047",
    "Participação ativa": "#1E88E5",
    "Sinais de atenção pedagógica": "#FB8C00",
    "Movimento e transição": "#7F8C8D",
}


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


def _behavior_share_map(summary_df: pd.DataFrame) -> dict[str, float]:
    if summary_df.empty:
        return {}
    return {
        str(row["behavior"]): float(row["duration_percentage"])
        for _, row in summary_df.iterrows()
    }


def _build_management_balance_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    behavior_share = _behavior_share_map(summary_df)
    rows = [
        {
            "group": "Engajamento na atividade",
            "share_percentage": round(
                behavior_share.get("Atento", 0.0) + behavior_share.get("Escrevendo", 0.0),
                2,
            ),
        },
        {
            "group": "Participação ativa",
            "share_percentage": round(behavior_share.get("Perguntando", 0.0), 2),
        },
        {
            "group": "Sinais de atenção pedagógica",
            "share_percentage": round(
                behavior_share.get("Distraído", 0.0)
                + behavior_share.get("Distraido", 0.0)
                + behavior_share.get("Dormindo", 0.0)
                + behavior_share.get("Agitado", 0.0),
                2,
            ),
        },
        {
            "group": "Movimento e transição",
            "share_percentage": round(behavior_share.get("Em Pé", 0.0), 2),
        },
    ]
    balance_df = pd.DataFrame(rows)
    balance_df = balance_df[balance_df["share_percentage"] > 0].copy()
    if balance_df.empty:
        return pd.DataFrame(columns=["group", "share_percentage"])
    return balance_df.sort_values("share_percentage", ascending=False).reset_index(drop=True)


def _signal_priority_label(share_percentage: float) -> str:
    if share_percentage >= 20:
        return "Alta prioridade"
    if share_percentage >= 8:
        return "Atenção moderada"
    return "Baixa prioridade"


def _build_signal_priority_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    behavior_share = _behavior_share_map(summary_df)
    rows = [
        {
            "signal": "Dispersão atencional",
            "share_percentage": round(
                behavior_share.get("Distraído", 0.0) + behavior_share.get("Distraido", 0.0),
                2,
            ),
            "guidance": "Reforçar instruções curtas, mediação por proximidade e retomadas de foco.",
        },
        {
            "signal": "Sonolência observada",
            "share_percentage": round(behavior_share.get("Dormindo", 0.0), 2),
            "guidance": "Verificar rotina, horário da aula e necessidade de acolhimento individual.",
        },
        {
            "signal": "Agitação e regulação",
            "share_percentage": round(behavior_share.get("Agitado", 0.0), 2),
            "guidance": "Planejar pausas, blocos curtos de tarefa e apoio socioemocional.",
        },
        {
            "signal": "Movimento fora da tarefa",
            "share_percentage": round(behavior_share.get("Em Pé", 0.0), 2),
            "guidance": "Observar gatilhos do deslocamento e revisar organização da atividade.",
        },
    ]
    signal_df = pd.DataFrame(rows)
    signal_df["priority"] = signal_df["share_percentage"].apply(_signal_priority_label)
    signal_df = signal_df[signal_df["share_percentage"] > 0].copy()
    if signal_df.empty:
        return pd.DataFrame(columns=["signal", "share_percentage", "priority", "guidance"])
    return signal_df.sort_values("share_percentage", ascending=False).reset_index(drop=True)


def _build_segment_heatmap_df(segment_distribution_df: pd.DataFrame) -> pd.DataFrame:
    if segment_distribution_df.empty:
        return pd.DataFrame()

    grouped = segment_distribution_df.copy()
    grouped["category"] = grouped["behavior"].map(
        {
            "Atento": "Engajamento na atividade",
            "Escrevendo": "Engajamento na atividade",
            "Perguntando": "Participação ativa",
            "Distraído": "Sinais de atenção pedagógica",
            "Distraido": "Sinais de atenção pedagógica",
            "Dormindo": "Sinais de atenção pedagógica",
            "Agitado": "Sinais de atenção pedagógica",
            "Em Pé": "Movimento e transição",
        }
    ).fillna("Movimento e transição")

    grouped = (
        grouped.groupby(["session_segment", "category"], as_index=False)["total_duration_seconds"]
        .sum()
        .rename(columns={"total_duration_seconds": "total_seconds"})
    )
    grouped["segment_total_seconds"] = grouped.groupby("session_segment")["total_seconds"].transform("sum")
    grouped["share_percentage"] = (
        grouped["total_seconds"] / grouped["segment_total_seconds"].clip(lower=1.0) * 100.0
    ).round(2)

    ordered_segments = ["Início da aula", "Meio da aula", "Final da aula"]
    ordered_categories = [
        "Engajamento na atividade",
        "Participação ativa",
        "Sinais de atenção pedagógica",
        "Movimento e transição",
    ]
    heatmap_df = (
        grouped.pivot(index="session_segment", columns="category", values="share_percentage")
        .reindex(index=ordered_segments, columns=ordered_categories)
        .fillna(0.0)
    )
    return heatmap_df


def _build_daily_management_trend_df(daily_distribution_df: pd.DataFrame) -> pd.DataFrame:
    if daily_distribution_df.empty:
        return pd.DataFrame(columns=["date", "date_label", "dimension", "share_percentage"])

    grouped = daily_distribution_df.copy()
    grouped["dimension"] = grouped["behavior"].map(
        {
            "Atento": "Engajamento",
            "Escrevendo": "Engajamento",
            "Perguntando": "Participação",
            "Distraído": "Sinais de atenção",
            "Distraido": "Sinais de atenção",
            "Dormindo": "Sinais de atenção",
            "Agitado": "Sinais de atenção",
            "Em Pé": "Movimento",
        }
    ).fillna("Movimento")
    grouped = (
        grouped.groupby(["date", "dimension"], as_index=False)["total_duration_seconds"]
        .sum()
        .rename(columns={"total_duration_seconds": "total_seconds"})
    )
    grouped["day_total_seconds"] = grouped.groupby("date")["total_seconds"].transform("sum")
    grouped["share_percentage"] = (
        grouped["total_seconds"] / grouped["day_total_seconds"].clip(lower=1.0) * 100.0
    ).round(2)
    grouped["date"] = pd.to_datetime(grouped["date"])
    grouped["date_label"] = grouped["date"].dt.strftime("%d/%m")
    return grouped.sort_values(["date", "dimension"]).reset_index(drop=True)


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
    if kind == "atencao_pedagogica":
        background = f"linear-gradient(180deg, {accent}18 0%, rgba(255,255,255,0.02) 100%)"
        shadow = f"0 10px 24px {accent}12, inset 0 1px 0 rgba(255,255,255,0.03)"
        min_height = "138px"
    else:
        background = "linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.015) 100%)"
        shadow = "inset 0 1px 0 rgba(255,255,255,0.03)"
        min_height = "132px"
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
            "<div style='padding: 0.78rem 0.88rem 0.74rem 0.88rem; border: 1px solid rgba(255,255,255,0.08); "
            f"border-left: 4px solid {accent}; border-radius: 0.95rem; background: {background}; "
            f"box-shadow: {shadow}; min-height: {min_height}; margin-bottom: 0.5rem;'>"
            f"<div style='font-size: 0.77rem; color: {accent}; margin-bottom: 0.3rem; font-weight: 600; letter-spacing: 0.01em;'>{html.escape(title)}</div>"
            f"<div style='font-size: 1.06rem; font-weight: 800; color: #F5F7FA; margin-bottom: 0.42rem;'>{html.escape(value)}</div>"
            f"{badge_markup}"
            f"<div style='font-size: 0.87rem; line-height: 1.38; color: #D8DEE7; text-align: left;'>{html.escape(supporting_text)}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _inject_report_top_styles() -> None:
    st.markdown(
        """
        <style>
        .report-top-hero {
            padding: 0.55rem 0 0.15rem 0;
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
            margin: 0.18rem 0 0.05rem 0;
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
            border-radius: 20px;
            border: 1px solid rgba(123, 144, 255, 0.18);
            background:
                radial-gradient(circle at top right, rgba(86, 112, 255, 0.18), transparent 34%),
                linear-gradient(180deg, rgba(29, 33, 48, 0.94) 0%, rgba(21, 24, 34, 0.98) 100%);
            padding: 0.9rem 0.95rem 0.95rem 0.95rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.18);
            margin: 0 0 0.2rem 0;
        }
        .report-action-align {
            margin-top: 3.8rem;
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
            font-size: 1.02rem;
            font-weight: 800;
            color: #F5F7FC;
            margin-bottom: 0.2rem;
        }
        .report-action-subtitle {
            font-size: 0.9rem;
            line-height: 1.5;
            color: #B6BED0;
            margin: 0;
        }
        .report-action-note {
            margin-top: 0.5rem;
            font-size: 0.84rem;
            color: #8F99AE;
        }
        .report-filter-note {
            margin-top: -0.15rem;
            margin-bottom: 0.65rem;
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
            padding: 0.78rem 0.9rem;
            margin-bottom: 0.45rem;
        }
        .overview-panel.executive {
            background: linear-gradient(180deg, rgba(56, 78, 112, 0.22) 0%, rgba(39, 49, 74, 0.16) 100%);
        }
        .overview-panel.interpretive {
            background: linear-gradient(180deg, rgba(255,255,255,0.035) 0%, rgba(255,255,255,0.02) 100%);
        }
        .overview-panel-title {
            font-size: 0.84rem;
            font-weight: 700;
            color: #F2F5FA;
            margin-bottom: 0.32rem;
            letter-spacing: 0.01em;
        }
        .overview-panel-body {
            color: #CDD4DF;
            line-height: 1.42;
            font-size: 0.93rem;
            max-width: 62ch;
        }
        .overview-panel-body.muted {
            color: #B3BBC8;
            font-size: 0.91rem;
        }
        .overview-summary-line {
            margin: 0.05rem 0 0.8rem 0;
            color: #E2E8F2;
            font-size: 1rem;
            line-height: 1.35;
            font-weight: 700;
            max-width: 76ch;
        }
        .overview-cards-shell {
            margin-bottom: 0.8rem;
        }
        .overview-interpretation-compact {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.07);
            background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.015) 100%);
            padding: 0.8rem 0.92rem;
            margin-bottom: 0.7rem;
        }
        .overview-interpretation-title {
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #AAB4C6;
            margin-bottom: 0.32rem;
        }
        .overview-interpretation-body {
            color: #D5DCE7;
            font-size: 0.92rem;
            line-height: 1.4;
            max-width: 74ch;
        }
        .attention-list {
            display: grid;
            gap: 0.5rem;
            margin-top: 0.2rem;
        }
        .attention-item {
            display: flex;
            gap: 0.55rem;
            align-items: flex-start;
            padding: 0.62rem 0.72rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(255,255,255,0.06);
        }
        .attention-marker {
            flex: 0 0 auto;
            color: #FFB74D;
            font-size: 0.92rem;
            line-height: 1;
            margin-top: 0.08rem;
        }
        .attention-text {
            color: #D4DAE4;
            line-height: 1.4;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_overview_summary_line(text: str) -> None:
    if not text:
        return
    st.markdown(f"<div class='overview-summary-line'>{html.escape(text)}</div>", unsafe_allow_html=True)


def _render_compact_interpretation(text: str) -> None:
    if not text:
        return
    st.markdown(
        (
            "<div class='overview-interpretation-compact'>"
            "<div class='overview-interpretation-title'>Leitura Pedagógica</div>"
            f"<div class='overview-interpretation-body'>{html.escape(text)}</div>"
            "</div>"
        ),
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

    st.markdown(
        """
        <div class="report-action-shell">
            <div class="report-action-eyebrow">Exportação</div>
            <div class="report-action-title">Baixar relatório e bases</div>
            <p class="report-action-subtitle">
                Exporte o PDF formal do recorte selecionado.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        "📄 Gerar PDF",
        data=pdf_bytes,
        file_name=f"relatorio_observacional_{export_base_name}.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary",
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

    body_class = "overview-panel-body" if variant in {"executive", "attention"} else "overview-panel-body muted"
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
            "<div class='attention-marker'>!</div>"
            f"<div class='attention-text'>{html.escape(concise_point)}</div>"
            "</div>"
        )

    st.markdown(f"<div class='attention-list'>{''.join(items)}</div>", unsafe_allow_html=True)


def _get_period_reference_label(report_data: dict) -> str:
    if report_data["period_mode"] == "Diário":
        return "a aula analisada"
    return "o período analisado"


def _get_predominant_consistency_row(report_data: dict):
    summary_df = report_data["behavior_summary"]
    consistency_df = report_data["behavior_consistency"]
    if summary_df.empty or consistency_df.empty:
        return None
    predominant_behavior = str(summary_df.iloc[0]["behavior"])
    match = consistency_df[consistency_df["behavior"] == predominant_behavior]
    if match.empty:
        return None
    return match.iloc[0]


def _map_regularidade_label(consistency_label: str | None) -> str:
    normalized = str(consistency_label or "").lower()
    if "regular" in normalized:
        return "Estável"
    if "episódica" in normalized or "episodica" in normalized:
        return "Oscilante"
    return "Variável"


def _detect_pedagogical_attention_signals(report_data: dict) -> dict[str, object]:
    summary_df = report_data["behavior_summary"]
    if summary_df.empty:
        return {"has_signal": False, "signals": [], "headline": "", "summary": ""}

    config = {
        "Distraído": {
            "display_name": "distração",
            "occurrence_threshold": 12.0,
            "duration_threshold": 10.0,
            "records_threshold": 3,
        },
        "Agitado": {
            "display_name": "agitação",
            "occurrence_threshold": 8.0,
            "duration_threshold": 8.0,
            "records_threshold": 2,
        },
        "Perguntando": {
            "display_name": "interação frequente",
            "occurrence_threshold": 18.0,
            "duration_threshold": 12.0,
            "records_threshold": 3,
        },
        "Dormindo": {
            "display_name": "sonolência observada",
            "occurrence_threshold": 6.0,
            "duration_threshold": 6.0,
            "records_threshold": 1,
        },
    }

    signals: list[dict[str, object]] = []
    for behavior_name, rules in config.items():
        row = summary_df[summary_df["behavior"] == behavior_name]
        if row.empty:
            continue
        row = row.iloc[0]
        occurrence = float(row["occurrence_percentage"])
        duration = float(row["duration_percentage"])
        records = int(row["records"])
        if (
            occurrence >= rules["occurrence_threshold"]
            or duration >= rules["duration_threshold"]
            or records >= rules["records_threshold"]
        ):
            signals.append(
                {
                    "behavior": behavior_name,
                    "display_name": rules["display_name"],
                    "occurrence_percentage": occurrence,
                    "duration_percentage": duration,
                    "records": records,
                }
            )

    signals.sort(
        key=lambda item: (
            item["occurrence_percentage"],
            item["duration_percentage"],
            item["records"],
        ),
        reverse=True,
    )

    if not signals:
        return {"has_signal": False, "signals": [], "headline": "", "summary": ""}

    main_signal = signals[0]
    if len(signals) == 1:
        headline = f"Sinal secundário de atenção pedagógica: {main_signal['behavior']}"
    else:
        other_behaviors = ", ".join(signal["behavior"] for signal in signals[1:3])
        headline = (
            f"Sinais secundários de atenção pedagógica: {main_signal['behavior']}"
            f" e {other_behaviors}"
        )

    if main_signal["behavior"] == "Perguntando":
        summary = (
            "Foram observadas ocorrências complementares de interação frequente. Esse dado não é, por si só, "
            "negativo, mas recomenda leitura articulada ao tipo de atividade e à dinâmica didática."
        )
    else:
        summary = (
            f"Apesar da predominância geral do período, foram observados episódios de {main_signal['display_name']} "
            "com presença suficiente para merecer observação contextualizada em aulas futuras."
        )

    return {
        "has_signal": True,
        "signals": signals,
        "headline": headline,
        "summary": summary,
    }


def _build_teacher_summary_text(report_data: dict) -> str:
    summary_df = report_data["behavior_summary"]
    metrics = report_data["headline_metrics"]
    if summary_df.empty:
        return "Não houve base suficiente para compor um resumo pedagógico do período."

    context_label = _get_period_reference_label(report_data)
    predominant_row = summary_df.iloc[0]
    consistency_row = _get_predominant_consistency_row(report_data)
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)

    predominant_behavior = str(predominant_row["behavior"]).lower()
    predominant_share = float(predominant_row["occurrence_percentage"])

    opening = f"Durante {context_label}, o comportamento observado foi predominantemente {predominant_behavior}."

    if pedagogical_signal["has_signal"]:
        return (
            f"{opening} Houve sinais secundários relevantes no período, o que recomenda leitura contextualizada "
            f"dos {metrics['total_records']} episódios analisados."
        )

    if consistency_row is not None and _map_regularidade_label(consistency_row["consistency_label"]) == "Estável":
        return (
            f"{opening} O padrão manteve estabilidade no recorte analisado, com {predominant_share:.1f}% dos registros associados a esse comportamento."
        )

    return (
        f"{opening} O padrão apresentou variações ao longo do recorte analisado, mantendo {predominant_share:.1f}% dos registros no comportamento predominante."
    )


def _build_overview_summary_line(report_data: dict) -> str:
    summary_df = report_data["behavior_summary"]
    if summary_df.empty:
        return "Sem base suficiente para síntese pedagógica."

    predominant_behavior = str(summary_df.iloc[0]["behavior"]).lower()
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)
    consistency_row = _get_predominant_consistency_row(report_data)
    regularidade = _map_regularidade_label(
        consistency_row["consistency_label"] if consistency_row is not None else None
    )

    if pedagogical_signal["has_signal"]:
        main_signal = str(pedagogical_signal["signals"][0]["display_name"])
        return f"Predominância de {predominant_behavior} com episódios de {main_signal} que requerem acompanhamento."

    if regularidade == "Estável":
        return f"Predominância de {predominant_behavior} com padrão estável no período analisado."

    return f"Predominância de {predominant_behavior} com variações observadas ao longo do período."


def _build_pedagogical_interpretation_text(report_data: dict) -> str:
    summary_df = report_data["behavior_summary"]
    if summary_df.empty:
        return "Não houve base suficiente para compor uma interpretação pedagógica do período."

    predominant_behavior = str(summary_df.iloc[0]["behavior"])
    consistency_row = _get_predominant_consistency_row(report_data)
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)
    regularidade = _map_regularidade_label(
        consistency_row["consistency_label"] if consistency_row is not None else None
    )
    behavior_lower = predominant_behavior.lower()

    if behavior_lower == "atento" and pedagogical_signal["has_signal"]:
        main_signal = pedagogical_signal["signals"][0]
        signal_name = str(main_signal["display_name"])
        return (
            f"As oscilações registradas, especialmente nos episódios compatíveis com {signal_name}, devem ser "
            "interpretadas à luz da dinâmica da aula e observadas em registros futuros."
        )

    if behavior_lower == "atento" and regularidade == "Estável":
        return (
            "O padrão observado é compatível com acompanhamento contínuo da atividade proposta, sem indícios "
            "observacionais de necessidade imediata de acompanhamento adicional."
        )

    if behavior_lower == "atento":
        return (
            "As variações registradas sugerem uma dinâmica menos estável e devem ser interpretadas em conjunto "
            "com o contexto da aula."
        )

    if behavior_lower in {"distraído", "distraido", "dormindo", "agitado"} and regularidade == "Estável":
        return (
            "O padrão observado foi recorrente ao longo do período e sugere acompanhamento em aulas futuras para "
            "verificar persistência e contexto de ocorrência."
        )

    if behavior_lower in {"distraído", "distraido", "dormindo", "agitado"}:
        return (
            "O padrão observado sugere necessidade de acompanhamento em aulas futuras, sempre com leitura "
            "articulada ao contexto pedagógico e sem interpretação diagnóstica."
        )

    return (
        "O padrão observado deve ser lido em conjunto com a atividade proposta e com o contexto pedagógico, "
        "priorizando a observação de consistência em registros futuros."
    )


def _build_teacher_highlights(report_data: dict) -> list[dict[str, str]]:
    summary_df = report_data["behavior_summary"]
    if summary_df.empty:
        return []

    comparison_lines = generate_previous_period_comparison(report_data)
    comparison_df = report_data["comparison"]
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)
    predominant_behavior = str(summary_df.iloc[0]["behavior"])
    consistency_row = _get_predominant_consistency_row(report_data)
    regularidade = _map_regularidade_label(
        consistency_row["consistency_label"] if consistency_row is not None else None
    )

    highlights = [
        {
            "kind": "predominancia",
            "title": "Predominância",
            "value": predominant_behavior,
            "supporting_text": "Comportamento com maior presença relativa no período analisado.",
        },
        {
            "kind": "padrao_geral",
            "title": "Regularidade do comportamento",
            "value": regularidade,
            "supporting_text": "Indica se o padrão principal apareceu de forma mais estável ou com oscilações ao longo do período.",
        },
    ]

    if pedagogical_signal["has_signal"]:
        main_signal = pedagogical_signal["signals"][0]
        highlights.append(
                {
                    "kind": "atencao_pedagogica",
                    "title": "Atenção pedagógica",
                    "value": main_signal["behavior"],
                    "supporting_text": "Comportamento secundário com presença suficiente para merecer observação pedagógica.",
                }
            )

    has_comparison_basis = (
        not comparison_df.empty and int(comparison_df["previous_records"].sum()) > 0 and comparison_lines
    )
    if has_comparison_basis:
        first_line = comparison_lines[0]
        if "Não há base" not in first_line and "Não houve registros suficientes" not in first_line:
            highlights.append(
                {
                    "kind": "variacao_principal",
                    "title": "Comparação recente",
                    "value": "Período anterior",
                    "supporting_text": _shorten_text(first_line, max_sentences=1),
                }
            )

    return highlights


def _order_teacher_highlights(highlights: list[dict[str, str]]) -> list[dict[str, str]]:
    priority = {
        "atencao_pedagogica": 0,
        "padrao_geral": 1,
        "predominancia": 2,
        "variacao_principal": 3,
    }
    return sorted(highlights, key=lambda item: priority.get(item["kind"], 99))


def _build_teacher_attention_points(report_data: dict) -> list[str]:
    summary_df = report_data["behavior_summary"]
    if summary_df.empty:
        return ["Não houve base suficiente para destacar pontos de atenção no período."]

    points: list[str] = []
    consistency_row = _get_predominant_consistency_row(report_data)
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)
    regularidade = _map_regularidade_label(
        consistency_row["consistency_label"] if consistency_row is not None else None
    )
    predominant_behavior = str(summary_df.iloc[0]["behavior"]).lower()

    if pedagogical_signal["has_signal"]:
        main_signal = pedagogical_signal["signals"][0]
        points.append(
            f"Monitorar a recorrência dos episódios de {str(main_signal['display_name'])} em aulas futuras."
        )
        points.append(
            "Verificar o contexto da aula nos momentos em que ocorreu a oscilação comportamental."
        )
        return points[:3]

    if predominant_behavior == "atento" and regularidade == "Estável":
        return ["Não foram observados padrões que indiquem necessidade de acompanhamento adicional no período analisado."]

    if regularidade in {"Variável", "Oscilante"} and len(points) < 2:
        points.append(
            "Reavaliar o padrão em novos registros para verificar se a oscilação se mantém."
        )

    if predominant_behavior in {"distraído", "distraido", "dormindo", "agitado"} and len(points) < 2:
        points.append(
            "Observar se o comportamento predominante se repete em outros contextos pedagógicos."
        )

    if not points:
        points.append("Manter acompanhamento rotineiro, sem necessidade de ampliação da leitura neste momento.")

    return points[:3]


def render_report_page(user_context: dict):
    _inject_report_top_styles()
    _render_report_top_header()

    base_options = get_available_filters(user_context)
    students_df = base_options["students"]
    if students_df.empty:
        st.warning("Não há episódios comportamentais registrados para gerar relatórios.")
        return

    with st.container(border=True):
        top_col_left, top_col_right = st.columns([2.25, 1], gap="medium")
        with top_col_left:
            st.markdown("<div class='report-panel-title'>Filtros do relatório</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='report-filter-note'>Defina o contexto de análise e, em seguida, utilize as ações de exportação ao lado.</div>",
                unsafe_allow_html=True,
            )

            controls_col1, controls_col2, controls_col3, controls_col4 = st.columns([1.2, 1.2, 1.2, 1.3], gap="small")
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
                student_placeholder = "Selecione um aluno"
                student_options = filtered_students_df["student"].tolist()
                if not student_options:
                    st.warning("Não há alunos com episódios para os filtros selecionados.")
                    return
                student_values = [student_placeholder] + student_options
                selected_student = st.selectbox("Aluno", student_values, index=0)

            controls_col5, controls_col6 = st.columns([0.9, 1.4], gap="small")
            with controls_col5:
                period_placeholder = "Selecione um período"
                period_values = [period_placeholder, "Diário", "Semanal", "Mensal"]
                period_mode = st.selectbox("Período", period_values, index=0)
            with controls_col6:
                today = date.today()
                if period_mode == period_placeholder:
                    st.date_input("Data de referência", value=today, disabled=True)
                    start_date = today
                    end_date = today
                else:
                    start_default, end_default = _default_range(period_mode)
                    student_last_date = (
                        _get_student_last_date(filtered_students_df, selected_student)
                        if selected_student != student_placeholder
                        else None
                    )
                    if period_mode == "Diário":
                        selected_date = st.date_input("Data de referência", value=today)
                        start_date = selected_date
                        end_date = selected_date
                    else:
                        selected_range = st.date_input(
                            "Intervalo de datas",
                            value=(
                                start_default,
                                today,
                            ),
                        )

                        if not isinstance(selected_range, (list, tuple)) or len(selected_range) != 2:
                            st.info("Selecione uma data inicial e uma data final para gerar o relatório.")
                            return

                        start_date, end_date = selected_range
        with top_col_right:
            top_actions_placeholder = st.empty()

    if selected_student == student_placeholder:
        st.info("Selecione um aluno para visualizar o relatório.")
        return

    if period_mode == period_placeholder:
        st.info("Selecione um período para visualizar o relatório.")
        return

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
    pedagogical_signal = _detect_pedagogical_attention_signals(report_data)
    overview_summary_line = _build_overview_summary_line(report_data)
    pedagogical_interpretation = _build_pedagogical_interpretation_text(report_data)
    teacher_highlights = _order_teacher_highlights(_build_teacher_highlights(report_data))
    teacher_attention_points = _build_teacher_attention_points(report_data)
    short_temporal_summary = _shorten_text(_build_timeline_temporal_summary(report_data), max_sentences=2)
    concise_comparison_lines = previous_period_lines[:3]

    with top_actions_placeholder.container():
        st.markdown("<div class='report-action-align'></div>", unsafe_allow_html=True)
        _render_report_export_actions(
            report_data=report_data,
            display_summary=display_summary,
            timeline_display=timeline_display,
            selected_student=selected_student,
            start_date=start_date,
            end_date=end_date,
        )

    predominant_share = float(summary_df.iloc[0]["duration_percentage"]) if not summary_df.empty else 0.0
    management_balance_df = _build_management_balance_df(summary_df)
    signal_priority_df = _build_signal_priority_df(summary_df)
    segment_heatmap_df = _build_segment_heatmap_df(report_data["session_segment_distribution"])
    daily_management_trend_df = _build_daily_management_trend_df(report_data["daily_distribution"])
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

    fig_management_balance = None
    if not management_balance_df.empty:
        fig_management_balance = px.pie(
            management_balance_df,
            values="share_percentage",
            names="group",
            hole=0.55,
            color="group",
            title="Balanço observacional do período",
            color_discrete_map=MANAGEMENT_CATEGORY_COLORS,
        )
        fig_management_balance.update_traces(
            texttemplate="%{value:.1f}%",
            textposition="inside",
        )
        fig_management_balance.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            legend_title_text="Leitura gerencial",
        )

    fig_signal_priority = None
    if not signal_priority_df.empty:
        fig_signal_priority = px.bar(
            signal_priority_df.sort_values("share_percentage", ascending=True),
            x="share_percentage",
            y="signal",
            orientation="h",
            color="priority",
            text="share_percentage",
            title="Sinais que mais pedem acompanhamento",
            labels={
                "share_percentage": "Percentual da duração (%)",
                "signal": "Sinal observacional",
                "priority": "Prioridade",
            },
            color_discrete_map=BEHAVIOR_SIGNAL_PRIORITY_COLORS,
            custom_data=["guidance"],
        )
        fig_signal_priority.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate=(
                "<b>Sinal:</b> %{y}<br>"
                "<b>Percentual:</b> %{x:.1f}%<br>"
                "<b>Leitura sugerida:</b> %{customdata[0]}"
                "<extra></extra>"
            ),
        )
        fig_signal_priority.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis=dict(ticksuffix="%"),
        )

    fig_timeline = _build_timeline_figure(timeline_df)

    fig_segment_heatmap = None
    if not segment_heatmap_df.empty:
        fig_segment_heatmap = go.Figure(
            data=go.Heatmap(
                z=segment_heatmap_df.values,
                x=list(segment_heatmap_df.columns),
                y=list(segment_heatmap_df.index),
                colorscale=[
                    [0.0, "#0B1F33"],
                    [0.35, "#1E88E5"],
                    [0.65, "#FBC02D"],
                    [1.0, "#E53935"],
                ],
                colorbar=dict(title="% no trecho"),
                hovertemplate=(
                    "<b>Momento da aula:</b> %{y}<br>"
                    "<b>Dimensão:</b> %{x}<br>"
                    "<b>Participação no trecho:</b> %{z:.1f}%<extra></extra>"
                ),
            )
        )
        fig_segment_heatmap.update_layout(
            title="Em que momento da aula os sinais aparecem",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Dimensão observada",
            yaxis_title="Trecho da aula",
        )

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

    fig_daily_management_trend = None
    if not daily_management_trend_df.empty and daily_management_trend_df["date"].nunique() > 1:
        fig_daily_management_trend = px.line(
            daily_management_trend_df,
            x="date",
            y="share_percentage",
            color="dimension",
            markers=True,
            title="Tendência observacional ao longo dos dias",
            labels={
                "date": "Data",
                "share_percentage": "Percentual da duração (%)",
                "dimension": "Dimensão",
            },
            color_discrete_map={
                "Engajamento": "#43A047",
                "Participação": "#1E88E5",
                "Sinais de atenção": "#FB8C00",
                "Movimento": "#7F8C8D",
            },
        )
        fig_daily_management_trend.update_layout(
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(ticksuffix="%"),
        )

    tabs = st.tabs(
        ["Visão Geral", "Indicadores Gerenciais", "Distribuição Temporal", "Comparações"]
    )

    with tabs[0]:
        st.subheader("Síntese Pedagógica")
        with st.container(border=True):
            _render_overview_summary_line(overview_summary_line)

            st.markdown("<div class='overview-cards-shell'>", unsafe_allow_html=True)
            if teacher_highlights:
                card_cols = st.columns(len(teacher_highlights), gap="medium")
                for idx, highlight in enumerate(teacher_highlights):
                    with card_cols[idx]:
                        _render_highlight_card(
                            highlight["kind"],
                            highlight["title"],
                            highlight["value"],
                            highlight["supporting_text"],
                        )
            else:
                st.info("Não houve destaques suficientes para compor este bloco.")
            st.markdown("</div>", unsafe_allow_html=True)

            _render_compact_interpretation(pedagogical_interpretation)

            st.markdown("##### Pontos de Atenção")
            _render_attention_points(teacher_attention_points)

    with tabs[1]:
        st.subheader("Indicadores Gerenciais")
        st.caption("Leitura visual voltada à gestão: equilíbrio geral, sinais prioritários e possíveis focos de intervenção.")
        chart_col1, chart_col2 = st.columns([1.05, 1.2], gap="large")
        with chart_col1:
            if fig_management_balance is not None:
                st.plotly_chart(fig_management_balance, width="stretch")
            else:
                st.info("Não houve base suficiente para consolidar o balanço observacional.")
        with chart_col2:
            if fig_signal_priority is not None:
                st.plotly_chart(fig_signal_priority, width="stretch")
            else:
                st.info("Não foram observados sinais com volume suficiente para priorização visual.")

        if not signal_priority_df.empty:
            st.markdown("##### Encaminhamentos sugeridos para acompanhamento")
            for _, row in signal_priority_df.head(3).iterrows():
                st.markdown(
                    f"- **{row['signal']}** ({row['priority']}): {row['guidance']}"
                )

        with st.expander("Ver distribuição detalhada por comportamento", expanded=False):
            st.caption("Visual detalhado de frequência e duração, útil para leitura técnica do caso.")
            chart_col1, chart_col2 = st.columns([1.2, 1], gap="large")
            with chart_col1:
                st.plotly_chart(fig_occurrence, width="stretch")
            with chart_col2:
                st.plotly_chart(fig_duration, width="stretch")
            st.dataframe(display_summary, width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("Distribuição Temporal")
        st.caption("Identifica em que momento da aula o padrão se intensifica e apoia decisões de intervenção pedagógica.")
        chart_col1, chart_col2 = st.columns([1.2, 1], gap="large")
        with chart_col1:
            st.plotly_chart(fig_timeline, width="stretch")
        with chart_col2:
            if fig_segment_heatmap is not None:
                st.plotly_chart(fig_segment_heatmap, width="stretch")
            else:
                st.info("Não houve base suficiente para comparar os trechos da aula.")
        st.info(short_temporal_summary or "Não houve base suficiente para sintetizar a leitura temporal.")

    with tabs[3]:
        st.subheader("Comparações")
        st.caption("Comparação com o período anterior e evolução do comportamento ao longo do recorte selecionado.")
        st.markdown("##### Síntese Comparativa")
        if concise_comparison_lines:
            for line in concise_comparison_lines:
                st.markdown(f"- {line}")
        else:
            st.info("Não há base comparativa suficiente para uma síntese resumida.")

        if fig_daily_management_trend is not None:
            st.markdown("##### Tendência por Dia")
            st.plotly_chart(fig_daily_management_trend, width="stretch")

        if fig_period is not None:
            st.markdown("##### Comparação entre Recortes do Período")
            st.plotly_chart(fig_period, width="stretch")

        with st.expander("Ver comparação detalhada", expanded=False):
            if comparison_display.empty:
                st.info("Não há base comparativa suficiente.")
            else:
                st.dataframe(comparison_display, width="stretch", hide_index=True)
