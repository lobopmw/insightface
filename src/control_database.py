
########################### ATUALIZAÇÃO ###########################################

try:
    import sqlite3
except Exception:
    sqlite3 = None

try:
    import streamlit as st
except Exception:
    st = None
## Imports de visualização movidos para dentro das funções
from contextlib import contextmanager
import pandas as pd
import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, UniqueConstraint, text
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
import os


import io
## Imports de PDF/imagem movidos para dentro das funções


import os  # se ainda não tiver

# Caminhos padrão (relativos à pasta src/)
HERE = os.path.dirname(__file__)
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
MAPPING_CSV = os.getenv("MAPPING_CSV", os.path.join(DATA_DIR, "mapeamento_alunos.csv"))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "behavior_data.db")

# Se variáveis de ambiente estiverem configuradas, usar PostgreSQL; caso contrário
# permanecer compatível com SQLite local. A migração já cria `control_database_postgres.py`,
# mas aqui oferecemos compatibilidade por ambiente.
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
else:
    DATABASE_URI = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URI, pool_pre_ping=True)

#-------------------------------------------------------------------------------------------------------------------------------------------------
@contextmanager
def connect_database():
    """Context manager para conectar ao banco configurado (SQLite ou PostgreSQL).

    Retorna uma tupla (connection, cursor) compatível com o código atual que usa
    operações SQL e pandas.read_sql_query.
    """
    # Se estiver usando SQLite, mantenha a compatibilidade com sqlite3
    if DATABASE_URI.startswith("sqlite:"):
        conn = sqlite3.connect(DB_PATH)
        try:
            cursor = conn.cursor()

            # Criar a tabela 'behavior_log' caso não exista (compatível com SQLite)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                school TEXT,
                discipline TEXT,
                teacher TEXT,
                student TEXT,
                id_student TEXT,
                behavior TEXT,
                count INTEGER DEFAULT 0,
                date TEXT,
                start_time TEXT,
                end_time TEXT,
                UNIQUE(student, behavior, date)
            )
            ''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT
            )
            ''')

            conn.commit()
            yield conn, cursor
        finally:
            conn.close()
    else:
        # PostgreSQL: usar SQLAlchemy engine
        with engine.connect() as conn:
            # O objeto `conn` do SQLAlchemy tem execute e pode ser usado pelo pandas
            # Garantir que as tabelas existam
            conn.execute(text('''
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
            '''))

            conn.execute(text('''
            CREATE TABLE IF NOT EXISTS students (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255)
            )
            '''))
            # Obter um cursor compatível com cursor.fetchall() usado em show_behavior_charts
            raw_conn = conn.connection
            # Commit via DBAPI connection
            try:
                raw_conn.commit()
            except Exception:
                pass
            cursor = raw_conn.cursor()
            try:
                yield raw_conn, cursor
            finally:
                try:
                    cursor.close()
                except:
                    pass
    


#------------------------------------------------------------------------------------------------------------------------------------------------
def insert_count_behavior(school, discipline, teacher, id_student, student, behavior, date, start_time, end_time, last_behavior=None):
    start_time = start_time or datetime.datetime.now().strftime("%H:%M:%S")
    end_time = end_time or datetime.datetime.now().strftime("%H:%M:%S")
    with connect_database() as (conn, cursor):
        # Detectar se estamos em SQLite (sqlite3 Cursor) ou PostgreSQL (psycopg2 cursor)
        is_sqlite = sqlite3 is not None and isinstance(cursor, sqlite3.Cursor)

        if last_behavior is None or last_behavior != behavior:
            if last_behavior is not None:
                if is_sqlite:
                    cursor.execute('''
                    UPDATE behavior_log
                    SET end_time = ?
                    WHERE id_student = ? AND student = ? AND behavior = ? AND school = ? AND discipline = ? AND teacher = ?
                    ''', (end_time, id_student, student, last_behavior, school, discipline, teacher))
                else:
                    cursor.execute('''
                    UPDATE behavior_log
                    SET end_time = %s
                    WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                    ''', (end_time, id_student, student, last_behavior, school, discipline, teacher))
                conn.commit()

            # Upsert (compatível com SQLite e PostgreSQL)
            if is_sqlite:
                cursor.execute('''
                INSERT INTO behavior_log (school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(student, behavior, date) DO UPDATE SET
                count = count + 1, start_time = COALESCE(start_time, ?), end_time = ?
                ''', (school, discipline, teacher, id_student, student, behavior, date, start_time, end_time, start_time, end_time))
            else:
                cursor.execute('''
                INSERT INTO behavior_log (school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time)
                VALUES (%s, %s, %s, %s, %s, %s, 1, %s, %s, %s)
                ON CONFLICT (student, behavior, date) 
                DO UPDATE SET
                    count = behavior_log.count + 1,
                    start_time = COALESCE(behavior_log.start_time, EXCLUDED.start_time),
                    end_time = EXCLUDED.end_time
                ''', (school, discipline, teacher, id_student, student, behavior, date, start_time, end_time))
            conn.commit()
        else:
            if is_sqlite:
                cursor.execute('''
                UPDATE behavior_log
                SET end_time = ?
                WHERE id_student = ? AND student = ? AND behavior = ? AND school = ? AND discipline = ? AND teacher = ?
                ''', (end_time, id_student, student, behavior, school, discipline, teacher))
            else:
                cursor.execute('''
                UPDATE behavior_log
                SET end_time = %s
                WHERE id_student = %s AND student = %s AND behavior = %s AND school = %s AND discipline = %s AND teacher = %s
                ''', (end_time, id_student, student, behavior, school, discipline, teacher))
            conn.commit()

    return behavior

#------------------------------------------------------------------------------------------------------------------------------------------------
def df_behavior_charts():
    query = """
    SELECT school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time 
    FROM behavior_log
    """
    with connect_database() as (conn, cursor):
        # conn pode ser sqlite3.Connection ou SQLAlchemy Connection; pandas aceita ambos,
        # porém para SQLAlchemy devemos passar a engine or connection
        try:
            df = pd.read_sql_query(query, conn)
        except Exception:
            # Caso conn seja um SQLAlchemy Connection, usar engine
            df = pd.read_sql_query(query, engine)

    if df.empty:
        df = pd.DataFrame(columns=[
            "school", "discipline", "teacher", "id_student", "student", 
            "behavior", "count", "date", "start_time", "end_time"
        ])
    
    df = df.rename(columns={
        'school': 'Escola',
        'discipline': 'Disciplina',
        'teacher': 'Professor',
        'id_student': 'Matrícula do Aluno',
        'student': 'Nome do Aluno',
        'behavior': 'Comportamento',
        'count': 'Nº Detecção',
        'date': 'Data',
        'start_time': 'Início',
        'end_time': 'Término'
    })

    return df

def show_behavior_charts():
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import plotly.express as px
    import plotly.graph_objects as go
    import io
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from PIL import Image
    import tempfile
    st.sidebar.header("Filtros")

    with connect_database() as (conn, cursor):
        is_sqlite = sqlite3 is not None and isinstance(cursor, sqlite3.Cursor)
        if is_sqlite:
            students = pd.read_sql_query("SELECT DISTINCT student FROM behavior_log", conn)['student'].tolist()
            disciplines = pd.read_sql_query("SELECT DISTINCT discipline FROM behavior_log", conn)['discipline'].tolist()
        else:
            students = pd.read_sql_query("SELECT DISTINCT student FROM behavior_log", engine)['student'].tolist()
            disciplines = pd.read_sql_query("SELECT DISTINCT discipline FROM behavior_log", engine)['discipline'].tolist()

    selected_student = st.sidebar.selectbox("Selecione um aluno", students, index=0)
    selected_discipline = st.sidebar.selectbox("Selecione a Disciplina", disciplines, index=0)

    selected_date = st.sidebar.date_input("Selecione a Data", value=datetime.datetime.today().date())
    selected_date = selected_date.strftime("%Y-%m-%d")
    data_formatada = datetime.datetime.strptime(selected_date, "%Y-%m-%d").strftime("%d/%m/%y")

    # Checar existência de dados usando conexão compatível
    with connect_database() as (conn, cursor):
        is_sqlite = sqlite3 is not None and isinstance(cursor, sqlite3.Cursor)
        if is_sqlite:
            query_check_date = f'''
                SELECT COUNT(*) FROM behavior_log
                WHERE student = ? AND discipline = ? AND date = ?
            '''
            cursor.execute(query_check_date, (selected_student, selected_discipline, selected_date))
            data_count = cursor.fetchone()[0]
        else:
            cursor.execute('''
                SELECT COUNT(*) FROM behavior_log
                WHERE student = %s AND discipline = %s AND date = %s
            ''', (selected_student, selected_discipline, selected_date))
            data_count = cursor.fetchone()[0]

    if data_count == 0:
        st.warning("Nenhum dado registrado para a data e disciplina selecionadas.")
        conn.close()
        return

    # Consulta de comportamento
    if is_sqlite:
        query_behavior = f'''
            SELECT behavior, SUM(count) as total_count
            FROM behavior_log
            WHERE student = ? AND discipline = ? AND date = ?
            GROUP BY behavior
        '''
        df_behavior = pd.read_sql_query(query_behavior, conn, params=(selected_student, selected_discipline, selected_date))
    else:
        query_behavior = f'''
            SELECT behavior, SUM(count) as total_count
            FROM behavior_log
            WHERE student = %s AND discipline = %s AND date = %s
            GROUP BY behavior
        '''
        df_behavior = pd.read_sql_query(query_behavior, engine, params=(selected_student, selected_discipline, selected_date))

    if is_sqlite:
        query_temporal = f'''
            SELECT behavior, start_time, end_time, SUM(count) as total_count
            FROM behavior_log
            WHERE student = ? AND discipline = ? AND date = ?
            GROUP BY behavior, start_time, end_time
            ORDER BY start_time
        '''
        df_temporal = pd.read_sql_query(query_temporal, conn, params=(selected_student, selected_discipline, selected_date))
    else:
        query_temporal = f'''
            SELECT behavior, start_time, end_time, SUM(count) as total_count
            FROM behavior_log
            WHERE student = %s AND discipline = %s AND date = %s
            GROUP BY behavior, start_time, end_time
            ORDER BY start_time
        '''
        df_temporal = pd.read_sql_query(query_temporal, engine, params=(selected_student, selected_discipline, selected_date))
    conn.close()

    if df_behavior.empty or df_temporal.empty:
        st.warning("Nenhum dado registrado para os filtros selecionados.")
        return

    st.title(f"Aluno: {selected_student}")

    cores = {
    "Atento": "royalblue",
    "Perguntando": "red",
    "Escrevendo": "orange",
    "Dormindo": "purple",
    "Agitado": "green",
    "Em Pé": "gray"
}

    fig_pie = px.pie(
        df_behavior, 
        values='total_count', 
        names='behavior', 
        title=f"Distribuição de Comportamentos - {selected_student} ({data_formatada})", 
        hole=0.3, 
        color_discrete_map=cores,
        template="plotly_white"
    )

    fig_bar = px.bar(
        df_behavior, 
        x='behavior', 
        y='total_count', 
        title=f"Contagem de Comportamentos - {selected_student} ({data_formatada})", 
        labels={'behavior': 'Comportamento', 'total_count': 'Quantidade'},
        color='behavior',
        text='total_count', 
        template='plotly_white'
    )
    fig_bar.update_traces(textposition='outside')

    df_temporal['start_time'] = pd.to_datetime(df_temporal['start_time'], format="%H:%M:%S")
    df_temporal['end_time'] = pd.to_datetime(df_temporal['end_time'], format="%H:%M:%S")

    min_time = df_temporal['start_time'].min()
    max_time = df_temporal['end_time'].max()
    time_intervals = pd.date_range(min_time, max_time, freq='10min')

    df_final = pd.DataFrame({'time': time_intervals})
    behaviors = df_temporal['behavior'].unique()

    for behavior in behaviors:
        cumulative_value = 0
        behavior_values = []

        for current_time in time_intervals:
            value = df_temporal[
                (df_temporal['behavior'] == behavior) &
                (df_temporal['start_time'] <= current_time) &
                (df_temporal['end_time'] >= current_time)
            ]['total_count'].sum()

            if value > cumulative_value:
                cumulative_value = value
            behavior_values.append(cumulative_value)

        df_final[behavior] = behavior_values

    df_final.fillna(0, inplace=True)
    df_final['time'] = pd.to_datetime(df_final['time'])
    df_final.set_index('time', inplace=True)
    horas_formatadas = df_final.index.strftime("%H:%M")

    fig_line = go.Figure()
    for column in df_final.columns:
        fig_line.add_trace(go.Scatter(
            x=horas_formatadas,
            y=df_final[column],
            mode='lines+markers',
            name=column, 
            line=dict(color=cores.get(column, 'black'))
        ))

    fig_line.update_layout(
        title=f"Evolução Temporal dos Comportamentos - {selected_student} ({data_formatada})",
        xaxis_title="Hora",
        yaxis_title="Contagem Acumulada",
        legend_title="Comportamentos",
        template="plotly_white"
    )

    def gerar_download_plotly(fig, nome_arquivo):
        buf = io.BytesIO()
        fig.write_image(buf, format='png')
        buf.seek(0)
        st.download_button(
            label=f"📥 Baixar gráfico: {nome_arquivo}",
            data=buf,
            file_name=f"{nome_arquivo}.png",
            mime="image/png"
        )
        return buf

    col_g1, col_g2, col_g3 = st.columns([2, 0.2, 2])
    with col_g1:
        st.plotly_chart(fig_pie, use_container_width=True)
        pie_buf = gerar_download_plotly(fig_pie, f"distribuicao_{selected_student}_{data_formatada}")

    with col_g3:
        st.plotly_chart(fig_bar, use_container_width=True)
        bar_buf = gerar_download_plotly(fig_bar, f"contagem_{selected_student}_{data_formatada}")

    st.plotly_chart(fig_line, use_container_width=True)
    line_buf = gerar_download_plotly(fig_line, f"evolucao_temporal_{selected_student}_{data_formatada}")

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
        c.drawImage(temp_image_path, 40, 100, width=500, preserveAspectRatio=True, mask='auto')
        c.showPage()
        os.unlink(temp_image_path)

    adicionar_pagina_pdf(pie_buf, "Distribuição de Comportamentos")
    adicionar_pagina_pdf(bar_buf, "Contagem de Comportamentos")
    adicionar_pagina_pdf(line_buf, "Evolução Temporal dos Comportamentos")

    c.save()
    pdf_buf.seek(0)

    st.download_button(
        label="📄 Baixar TODOS os gráficos em PDF",
        data=pdf_buf,
        file_name=f"graficos_{selected_student}_{selected_date}.pdf",
        mime="application/pdf"
    )

#----------------------------------- Tabela do usuário -------------------------------------------------------------------------------------------
def user_table():
    try:
        metadata = MetaData()
        users = Table(
            'users', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('cpf', String(11), unique=True, nullable=False),
            Column('nome', String(255), nullable=False),
            Column('password', String, nullable=False),
            Column('cidade', String, nullable=False),
            Column('estado', String(2), nullable=False),
            UniqueConstraint('cpf', name='uix_1')
        )
        metadata.create_all(engine)
        print("Tabela 'users' criada/verificada com sucesso.")
    except SQLAlchemyError as e:
        print(f"Erro ao criar/verificar tabela 'users': {e}")

#------------------------------------------------------------------------------------------------------------------------------------------------
def registrar_usuario(cpf, nome, hashed_password, cidade, estado):
    try:
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) FROM users WHERE cpf = :cpf")
            result = conn.execute(query, {"cpf": cpf}).scalar()

            if result > 0:
                print("CPF já existe no banco de dados.")
                return False

            insert_query = text("""
                INSERT INTO users (cpf, nome, password, cidade, estado)
                VALUES (:cpf, :nome, :password, :cidade, :estado)
            """)
            conn.execute(insert_query, {
                "cpf": cpf,
                "nome": nome,
                "password": hashed_password,
                "cidade": cidade,
                "estado": estado
            })
            conn.commit()
            print(f"Usuário {nome} registrado com sucesso.")
        return True
    except SQLAlchemyError as e:
        print(f"Erro ao registrar usuário: {e}")
        return False
