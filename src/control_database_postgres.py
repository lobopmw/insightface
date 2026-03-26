########################### VERSAO POSTGRESQL ###########################################

import os
import streamlit as st
from contextlib import contextmanager
import pandas as pd
import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, UniqueConstraint, text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import plotly.express as px
import psycopg2
from pgvector.psycopg2 import register_vector

import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from PIL import Image
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Configuracoes do banco de dados
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "insightface_db")
DB_USER = os.getenv("DB_USER", "insightface_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "insightface_password")

# URI de conexao PostgreSQL
DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Engine SQLAlchemy
engine = create_engine(
    DATABASE_URI,
    pool_pre_ping=True,
    pool_recycle=300,
    pool_size=10,
    max_overflow=20,
)


@contextmanager
def connect_database():
    """Context manager para conexao PostgreSQL."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    original_autocommit = conn.autocommit
    try:
        try:
            conn.autocommit = True
            with conn.cursor() as _cursor:
                _cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        finally:
            conn.autocommit = original_autocommit

        register_vector(conn)
        cursor = conn.cursor()

        # Tabela legado: contagem por frame/transicao
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS behavior_log (
                id SERIAL PRIMARY KEY,
                school VARCHAR(255),
                discipline VARCHAR(255),
                teacher VARCHAR(255),
                student VARCHAR(255),
                id_student VARCHAR(255),
                behavior VARCHAR(100),
                count INTEGER DEFAULT 0,
                date VARCHAR(10),
                start_time VARCHAR(8),
                end_time VARCHAR(8),
                CONSTRAINT uq_behavior_log UNIQUE(student, behavior, date)
            )
            '''
        )

        # Nova tabela: episodios de comportamento (tempo)
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS behavior_episode (
                id SERIAL PRIMARY KEY,
                school VARCHAR(255),
                discipline VARCHAR(255),
                teacher VARCHAR(255),
                student VARCHAR(255) NOT NULL,
                id_student VARCHAR(255),
                behavior VARCHAR(100) NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                duration_seconds DOUBLE PRECISION NOT NULL,
                date DATE NOT NULL,
                source VARCHAR(20) DEFAULT 'realtime'
            )
            '''
        )

        cursor.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_date
            ON behavior_episode (student, date)
            '''
        )

        cursor.execute(
            '''
            CREATE INDEX IF NOT EXISTS idx_behavior_episode_student_start_time
            ON behavior_episode (student, start_time)
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS students (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255)
            )
            '''
        )

        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                student_hash VARCHAR(64) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                matricula VARCHAR(255),
                embedding vector(512) NOT NULL,
                updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )

        conn.commit()
        yield conn, cursor
    finally:
        conn.close()


# ------------------------------------------------------------------------------------------------------------------------------------------------
def insert_count_behavior(
    school,
    discipline,
    teacher,
    id_student,
    student,
    behavior,
    date,
    start_time,
    end_time,
    last_behavior=None,
):
    """LEGADO: Inserir/atualizar comportamento no behavior_log."""
    start_time = start_time or datetime.datetime.now().strftime("%H:%M:%S")
    end_time = end_time or datetime.datetime.now().strftime("%H:%M:%S")

    with connect_database() as (conn, cursor):
        if last_behavior is None or last_behavior != behavior:
            if last_behavior is not None:
                cursor.execute(
                    '''
                    UPDATE behavior_log
                    SET end_time = %s
                    WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                    ''',
                    (end_time, id_student, student, last_behavior, school, discipline, teacher),
                )
                conn.commit()

            cursor.execute(
                '''
                INSERT INTO behavior_log (school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
                ON CONFLICT (student, behavior, date)
                DO UPDATE SET
                    count = behavior_log.count + 1,
                    start_time = COALESCE(behavior_log.start_time, EXCLUDED.start_time),
                    end_time = EXCLUDED.end_time
                ''',
                (school, discipline, teacher, id_student, student, behavior, date, start_time, end_time),
            )
            conn.commit()
        else:
            cursor.execute(
                '''
                UPDATE behavior_log
                SET end_time = %s
                WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                ''',
                (end_time, id_student, student, behavior, school, discipline, teacher),
            )
            conn.commit()

    return behavior


# ------------------------------------------------------------------------------------------------------------------------------------------------
def insert_behavior_episode(
    school,
    discipline,
    teacher,
    id_student,
    student,
    behavior,
    start_time,
    end_time,
    source="realtime",
):
    """Persistir episodio consolidado de comportamento."""
    if not student or not behavior or start_time is None or end_time is None:
        return

    if end_time <= start_time:
        return

    duration_seconds = float((end_time - start_time).total_seconds())
    if duration_seconds <= 0:
        return

    with connect_database() as (conn, cursor):
        cursor.execute(
            '''
            INSERT INTO behavior_episode (
                school, discipline, teacher, id_student, student, behavior,
                start_time, end_time, duration_seconds, date, source
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''',
            (
                school,
                discipline,
                teacher,
                id_student,
                student,
                behavior,
                start_time,
                end_time,
                duration_seconds,
                start_time.date(),
                source,
            ),
        )
        conn.commit()


# ------------------------------------------------------------------------------------------------------------------------------------------------
def df_behavior_charts():
    """Obter episodios para tabela no Streamlit."""
    query = """
    SELECT school, discipline, teacher, id_student, student, behavior,
           date, start_time, end_time, duration_seconds, source
    FROM behavior_episode
    ORDER BY start_time DESC
    """
    with connect_database() as (conn, _cursor):
        df = pd.read_sql_query(query, conn)

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "school",
                "discipline",
                "teacher",
                "id_student",
                "student",
                "behavior",
                "date",
                "start_time",
                "end_time",
                "duration_seconds",
                "source",
            ]
        )

    df = df.rename(
        columns={
            "school": "Escola",
            "discipline": "Disciplina",
            "teacher": "Professor",
            "id_student": "Matrícula do Aluno",
            "student": "Nome do Aluno",
            "behavior": "Comportamento",
            "date": "Data",
            "start_time": "Início",
            "end_time": "Término",
            "duration_seconds": "Duração (s)",
            "source": "Origem",
        }
    )

    return df


# -------------------------------- Gráficos com nome somente o primeiro nome do aluno ----------------------------------------
def show_behavior_charts():
    """Mostrar graficos de comportamento baseados em episodios."""
    with connect_database() as (conn, cursor):
        st.sidebar.header("Filtros")

        cursor.execute("SELECT DISTINCT student FROM behavior_episode ORDER BY student")
        students = [row[0] for row in cursor.fetchall()]

        if not students:
            st.warning("Sem dados de episódios comportamentais para exibir.")
            return

        selected_student = st.sidebar.selectbox("Selecione um aluno", students, index=0)

        cursor.execute("SELECT DISTINCT discipline FROM behavior_episode ORDER BY discipline")
        disciplines = [row[0] for row in cursor.fetchall() if row[0]]
        selected_discipline = (
            st.sidebar.selectbox("Selecione a Disciplina", disciplines, index=0) if disciplines else None
        )

        selected_date = st.sidebar.date_input("Selecione a Data", value=datetime.datetime.today().date())
        selected_date_str = selected_date.strftime("%Y-%m-%d")
        data_formatada = datetime.datetime.strptime(selected_date_str, "%Y-%m-%d").strftime("%d/%m/%y")

        if selected_discipline:
            cursor.execute(
                '''
                SELECT COUNT(*) FROM behavior_episode
                WHERE student = %s AND discipline = %s AND date = %s
                ''',
                (selected_student, selected_discipline, selected_date_str),
            )
        else:
            cursor.execute(
                '''
                SELECT COUNT(*) FROM behavior_episode
                WHERE student = %s AND date = %s
                ''',
                (selected_student, selected_date_str),
            )

        data_count = cursor.fetchone()[0]
        if data_count == 0:
            st.warning("Sem dados para os filtros selecionados.")
            return

        if selected_discipline:
            cursor.execute(
                '''
                SELECT behavior, SUM(duration_seconds) AS total_seconds
                FROM behavior_episode
                WHERE student = %s AND discipline = %s AND date = %s
                GROUP BY behavior
                ''',
                (selected_student, selected_discipline, selected_date_str),
            )
            df_behavior = pd.DataFrame(cursor.fetchall(), columns=["behavior", "total_seconds"])

            cursor.execute(
                '''
                SELECT behavior, start_time, end_time, duration_seconds
                FROM behavior_episode
                WHERE student = %s AND discipline = %s AND date = %s
                ORDER BY start_time
                ''',
                (selected_student, selected_discipline, selected_date_str),
            )
        else:
            cursor.execute(
                '''
                SELECT behavior, SUM(duration_seconds) AS total_seconds
                FROM behavior_episode
                WHERE student = %s AND date = %s
                GROUP BY behavior
                ''',
                (selected_student, selected_date_str),
            )
            df_behavior = pd.DataFrame(cursor.fetchall(), columns=["behavior", "total_seconds"])

            cursor.execute(
                '''
                SELECT behavior, start_time, end_time, duration_seconds
                FROM behavior_episode
                WHERE student = %s AND date = %s
                ORDER BY start_time
                ''',
                (selected_student, selected_date_str),
            )

        df_temporal = pd.DataFrame(
            cursor.fetchall(), columns=["behavior", "start_time", "end_time", "duration_seconds"]
        )

    if df_behavior.empty or df_temporal.empty:
        st.warning("Nenhum dado registrado para os filtros selecionados.")
        return

    st.title(f"Aluno: {selected_student}")
    df_behavior["total_minutes"] = (df_behavior["total_seconds"] / 60.0).round(1)

    cores = {
        "Atento": "royalblue",
        "Perguntando": "red",
        "Escrevendo": "orange",
        "Dormindo": "purple",
        "Agitado": "green",
        "Em Pé": "gray",
        "Distraido": "brown",
    }

    fig_pie = px.pie(
        df_behavior,
        values="total_minutes",
        names="behavior",
        title=f"Percentual do Tempo por Comportamento - {selected_student} ({data_formatada})",
        hole=0.4,
        color_discrete_map=cores,
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        template="plotly_white",
    )
    fig_pie.update_layout(legend_title_text="Comportamento")
    fig_pie.update_traces(
        texttemplate="%{percent}",
        hovertemplate="Comportamento=%{label}<br>Tempo (minutos)=%{value:.1f}<br>Percentual=%{percent}<extra></extra>",
    )

    fig_bar = px.bar(
        df_behavior,
        x="behavior",
        y="total_minutes",
        title=f"Tempo Total por Comportamento (min) - {selected_student} ({data_formatada})",
        labels={"behavior": "Comportamento", "total_minutes": "Tempo (minutos)"},
        color="behavior",
        text="total_minutes",
        template="plotly_white",
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_bar.update_layout(legend_title_text="Comportamento")
    fig_bar.update_yaxes(tickformat=".0f")
    fig_bar.update_traces(hovertemplate="Comportamento=%{x}<br>Tempo (minutos)=%{y:.1f}<extra></extra>")

    df_temporal["start_time"] = pd.to_datetime(df_temporal["start_time"])
    df_temporal["end_time"] = pd.to_datetime(df_temporal["end_time"])

    fig_timeline = px.timeline(
        df_temporal,
        x_start="start_time",
        x_end="end_time",
        y="behavior",
        color="behavior",
        color_discrete_map=cores,
        title=f"Linha do Tempo dos Episodios - {selected_student} ({data_formatada})",
        labels={"behavior": "Comportamento", "start_time": "Inicio", "end_time": "Fim"},
        template="plotly_white",
    )
    fig_timeline.update_yaxes(autorange="reversed")
    fig_timeline.update_layout(legend_title_text="Comportamento")
    fig_timeline.update_xaxes(
        showgrid=True,
        gridcolor="rgba(150,150,150,0.35)",
        gridwidth=1,
        tickformat="%H:%M<br>%d/%m/%Y",
        hoverformat="%d/%m/%Y %H:%M:%S",
    )
    fig_timeline.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.25)", gridwidth=1)

    def gerar_download_plotly(fig, nome_arquivo):
        buf = io.BytesIO()
        try:
            fig.write_image(buf, format="png")
        except Exception:
            return None
        buf.seek(0)
        st.download_button(
            label=f"📥 Baixar gráfico: {nome_arquivo}",
            data=buf,
            file_name=f"{nome_arquivo}.png",
            mime="image/png",
        )
        return buf

    plotly_config_base = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    col_g1, col_g2, col_g3 = st.columns([2, 0.2, 2])
    with col_g1:
        pie_config = dict(plotly_config_base)
        pie_config["toImageButtonOptions"] = {"format": "png", "filename": f"distribuicao_{selected_student}_{data_formatada}", "scale": 2}
        st.plotly_chart(fig_pie, use_container_width=True, config=pie_config)
        pie_buf = gerar_download_plotly(fig_pie, f"distribuicao_{selected_student}_{data_formatada}")

    with col_g3:
        bar_config = dict(plotly_config_base)
        bar_config["toImageButtonOptions"] = {"format": "png", "filename": f"tempo_total_{selected_student}_{data_formatada}", "scale": 2}
        st.plotly_chart(fig_bar, use_container_width=True, config=bar_config)
        bar_buf = gerar_download_plotly(fig_bar, f"tempo_total_{selected_student}_{data_formatada}")

    timeline_config = dict(plotly_config_base)
    timeline_config["toImageButtonOptions"] = {"format": "png", "filename": f"timeline_{selected_student}_{data_formatada}", "scale": 2}
    st.plotly_chart(fig_timeline, use_container_width=True, config=timeline_config)
    timeline_buf = gerar_download_plotly(fig_timeline, f"timeline_{selected_student}_{data_formatada}")

    if pie_buf is None or bar_buf is None or timeline_buf is None:
        st.info("Download PNG pelo botão de câmera do próprio gráfico está disponível.")
        return

    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)

    def adicionar_pagina_pdf(buffer_img, titulo):
        buffer_img.seek(0)
        img = Image.open(buffer_img)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            img.convert("RGB").save(tmp_file.name)
            temp_image_path = tmp_file.name

        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, 800, f"{titulo} - Aluno: {selected_student} - Data: {data_formatada}")
        c.drawImage(temp_image_path, 40, 100, width=500, preserveAspectRatio=True, mask="auto")
        c.showPage()
        os.unlink(temp_image_path)

    adicionar_pagina_pdf(pie_buf, "Percentual do Tempo por Comportamento")
    adicionar_pagina_pdf(bar_buf, "Tempo Total por Comportamento")
    adicionar_pagina_pdf(timeline_buf, "Linha do Tempo dos Episodios")

    c.save()
    pdf_buf.seek(0)

    st.download_button(
        label="📄 Baixar TODOS os gráficos em PDF",
        data=pdf_buf,
        file_name=f"graficos_{selected_student}_{selected_date_str}.pdf",
        mime="application/pdf",
    )


# ----------------------------------- Tabela do usuario -------------------------------------------------------------------------------------------
def user_table():
    """Criar tabela de usuarios no PostgreSQL"""
    try:
        metadata = MetaData()
        users = Table(
            "users",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("cpf", String(11), unique=True, nullable=False),
            Column("nome", String(255), nullable=False),
            Column("password", String(255), nullable=False),
            Column("cidade", String(255), nullable=False),
            Column("estado", String(2), nullable=False),
            UniqueConstraint("cpf", name="uq_users_cpf"),
        )
        metadata.create_all(engine)
        print("✅ Tabela 'users' criada/verificada com sucesso no PostgreSQL.")
    except SQLAlchemyError as e:
        print(f"❌ Erro ao criar/verificar tabela 'users': {e}")


# ------------------------------------------------------------------------------------------------------------------------------------------------
def registrar_usuario(cpf, nome, hashed_password, cidade, estado):
    """Registrar usuario no PostgreSQL"""
    try:
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) FROM users WHERE cpf = :cpf")
            result = conn.execute(query, {"cpf": cpf}).scalar()

            if result > 0:
                print("CPF ja existe no banco de dados.")
                return "cpf_exists"

            insert_query = text(
                """
                INSERT INTO users (cpf, nome, password, cidade, estado)
                VALUES (:cpf, :nome, :password, :cidade, :estado)
                """
            )
            conn.execute(
                insert_query,
                {
                    "cpf": cpf,
                    "nome": nome,
                    "password": hashed_password,
                    "cidade": cidade,
                    "estado": estado,
                },
            )
            conn.commit()
            print(f"✅ Usuario {nome} registrado com sucesso no PostgreSQL.")
        return "ok"
    except IntegrityError as e:
        print(f"❌ Erro de integridade ao registrar usuario: {e}")
        return "cpf_exists"
    except SQLAlchemyError as e:
        print(f"❌ Erro ao registrar usuario: {e}")
        return "error"
