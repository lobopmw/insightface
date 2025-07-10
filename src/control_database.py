
########################### ATUALIZAÇÃO ###########################################

import sqlite3
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from contextlib import contextmanager
import pandas as pd
import datetime
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, UniqueConstraint, text
from sqlalchemy.exc import SQLAlchemyError
import plotly.express as px
import plotly.graph_objects as go
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "..", "model")
os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "behavior_data.db")
DATABASE_URI = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URI)

#-------------------------------------------------------------------------------------------------------------------------------------------------
@contextmanager
def connect_database():
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()

        # Criar a tabela 'behavior_log'
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

        # Criar a tabela 'students'
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
    


#------------------------------------------------------------------------------------------------------------------------------------------------
def insert_count_behavior(school, discipline, teacher, id_student, student, behavior, date, start_time, end_time, last_behavior=None):
    start_time = start_time or datetime.datetime.now().strftime("%H:%M:%S")
    end_time = end_time or datetime.datetime.now().strftime("%H:%M:%S")

    with connect_database() as (conn, cursor):
        if last_behavior is None or last_behavior != behavior:
            if last_behavior is not None:
                cursor.execute('''
                UPDATE behavior_log
                SET end_time = ?
                WHERE id_student = ? AND student = ? AND behavior = ? AND school = ? AND discipline = ? AND teacher = ?
                ''', (end_time, id_student, student, last_behavior, school, discipline, teacher))
                conn.commit()

            cursor.execute('''
            INSERT INTO behavior_log (school, discipline, teacher, id_student, student, behavior, count, date, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(student, behavior, date) DO UPDATE SET
            count = count + 1, start_time = COALESCE(start_time, ?), end_time = ?
            ''', (school, discipline, teacher, id_student, student, behavior, date, start_time, end_time, start_time, end_time))
            conn.commit()
        else:
            cursor.execute('''
            UPDATE behavior_log
            SET end_time = ?
            WHERE id_student = ? AND student = ? AND behavior = ? AND school = ? AND discipline = ? AND teacher = ?
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
        df = pd.read_sql_query(query, conn)

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

#------------------------------------------------------------------------------------------------------------------------------------------------
def show_behavior_charts():
    conn = sqlite3.connect(DB_PATH)
    st.sidebar.header("Filtros")

    students = pd.read_sql_query("SELECT DISTINCT student FROM behavior_log", conn)['student'].tolist()
    selected_student = st.sidebar.selectbox("Selecione um aluno", students, index=0)

    disciplines = pd.read_sql_query("SELECT DISTINCT discipline FROM behavior_log", conn)['discipline'].tolist()
    selected_discipline = st.sidebar.selectbox("Selecione a Disciplina", disciplines, index=0)

    selected_date = st.sidebar.date_input("Selecione a Data", value=datetime.datetime.today().date())
    selected_date = selected_date.strftime("%Y-%m-%d")

    query_check_date = f'''
        SELECT COUNT(*) FROM behavior_log
        WHERE student = "{selected_student}" AND discipline = "{selected_discipline}" AND date = "{selected_date}"
    '''
    cursor = conn.cursor()
    cursor.execute(query_check_date)
    data_count = cursor.fetchone()[0]

    if data_count == 0:
        st.warning("Nenhum dado registrado para a data e disciplina selecionadas.")
        conn.close()
        return

    query_behavior = f'''
        SELECT behavior, SUM(count) as total_count
        FROM behavior_log
        WHERE student = "{selected_student}" AND discipline = "{selected_discipline}" AND date = "{selected_date}"
        GROUP BY behavior
    '''
    df_behavior = pd.read_sql_query(query_behavior, conn)

    query_temporal = f'''
        SELECT behavior, start_time, end_time, SUM(count) as total_count
        FROM behavior_log
        WHERE student = "{selected_student}" AND discipline = "{selected_discipline}" AND date = "{selected_date}"
        GROUP BY behavior, start_time, end_time
        ORDER BY start_time
    '''
    df_temporal = pd.read_sql_query(query_temporal, conn)
    conn.close()

    if df_behavior.empty or df_temporal.empty:
        st.warning("Nenhum dado registrado para os filtros selecionados.")
        return

    st.title(f"Aluno: {selected_student}")

    fig_pie = px.pie(
        df_behavior, 
        values='total_count', 
        names='behavior', 
        title='Distribuição de Comportamentos', 
        hole=0.3
    )

    fig_bar = px.bar(
        df_behavior, 
        x='behavior', 
        y='total_count', 
        title='Contagem de Comportamentos', 
        labels={'behavior': 'Comportamento', 'total_count': 'Quantidade'},
        color='behavior',
        text='total_count'
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
    df_final.set_index('time', inplace=True)

    fig_line = go.Figure()
    for column in df_final.columns:
        fig_line.add_trace(go.Scatter(
            x=df_final.index,
            y=df_final[column],
            mode='lines+markers',
            name=column
        ))

    fig_line.update_layout(
        title="Evolução Temporal dos Comportamentos",
        xaxis_title="Tempo",
        yaxis_title="Contagem Acumulada",
        xaxis=dict(tickformat="%H:%M"),
        legend_title="Comportamentos",
        template="plotly_white"
    )

    col_g1, col_g2, col_g3 = st.columns([2, 1, 2])
    with col_g1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_g3:
        st.plotly_chart(fig_bar, use_container_width=True)

    st.plotly_chart(fig_line, use_container_width=True)

#------------------------------------------------------------------------------------------------------------------------------------------------
def user_table():
    try:
        metadata = MetaData()
        users = Table(
            'users', metadata,
            Column('id', Integer, primary_key=True, autoincrement=True),
            Column('cpf', String(11), unique=True, nullable=False),
            Column('nome', String, nullable=False),
            Column('password', String, nullable=False),
            Column('cidade', String, nullable=False),
            Column('estado', String, nullable=False),
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
