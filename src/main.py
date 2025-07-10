
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import bcrypt
from sqlalchemy import text
from control_database import engine, registrar_usuario, user_table
from streamlit_cookies_controller import CookieController
from insightface_classroom import recognition_behavior
from register_face_multi_images_avg import register_faces
import os


#-----------------------------------------------------------------------------------------------------------------------------------#
# Configurações iniciais
st.set_page_config(page_title="Monitoramento - SEDUC", page_icon="../images/icon_school.jpg", layout="wide")

image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))

# Criando a tablea usuário
user_table()


# Funções de manipulação de "cookies" usando query params
def set_cookie(key, value):
   
    st.query_params[key] = value  # Define diretamente no query_params

def get_cookie(key):
    
    return st.query_params.get(key, None)  # Retorna o valor ou None

def delete_cookie(key):
   
    if key in st.query_params:
        del st.query_params[key]  # Remove o query_param correspondente


# Inicialização do estado da sessão
if "authenticated" not in st.session_state:
    # Restaurar estado a partir dos query params
    if get_cookie("authenticated") == "true":
        st.session_state['authenticated'] = True
        st.session_state['cpf'] = get_cookie("cpf")
        st.session_state['city'] = get_cookie("city")
        st.session_state['state'] = get_cookie("state")
        st.session_state['name'] = get_cookie("name")

    else:
        st.session_state["authenticated"] = False
        st.session_state['cpf'] = None
        st.session_state['city'] = None
        st.session_state['state'] = None
        st.session_state['name'] = None


#----------------------------------------------------------------------------------------------------------------------------------------#
# Função Login

def login():
    
    # Coleta de informações de login
    colbutton1,colbutton2,colbutton3 = st.columns([1,3,1])
    with colbutton2:
        cpf = st.text_input("CPF", placeholder="👤 CPF", max_chars=11, label_visibility= "hidden")
        password = st.text_input("Senha", type="password", placeholder="🔒 Senha", label_visibility= "hidden")

    with colbutton2:

        if st.button("**➡ Login**", use_container_width=True, key="submit-button"):
            
            if validar_cpf(cpf):
                try:
                    with engine.connect() as conn:
                        query = text("SELECT nome, cpf, password, cidade, estado FROM users WHERE cpf = :cpf")
                        result = conn.execute(query, {"cpf": cpf}).fetchone()

                        if result:
                            stored_nome, stored_cpf, stored_password, stored_city, stored_state = result
                            if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
                                st.session_state['authenticated'] = True
                                st.session_state['name'] = stored_nome
                                st.session_state['cpf'] = stored_cpf
                                st.session_state['city'] = stored_city
                                st.session_state['state'] = stored_state
                                
                                # Salvar estado nos "cookies"
                                set_cookie("authenticated", "true")
                                set_cookie("name", stored_nome)
                                set_cookie("cpf", stored_cpf)
                                set_cookie("city", stored_city)
                                set_cookie("state", stored_state)

                                st.success(f"Login realizado com sucesso! Bem-vindo, {stored_nome}")
                                st.rerun()
                            else:
                                st.error("Usuário ou senha incorretos!")
                        else:
                            st.error("Usuário não encontrado!")
                except Exception as e:
                    st.error(f"Erro ao validar login: {e}")
            else:
                st.warning("CPF inválido!")
    #     # Link para a recuperação de senha
    # st.markdown(
    #     "<a style='display: block; text-align: center; color: blue;' href='#' >Esqueci minha senha</a>",
    #     unsafe_allow_html=True,
    # )
    
#-----------------------------------------------------------------------------------------------------------------------------------------#
#Função Cadastrar

# Criando uma lista de estados
estados = [
    {"nome": "Informe o estado", "sigla": "BR"},
    {"nome": "Acre", "sigla": "AC"},
    {"nome": "Alagoas", "sigla": "AL"},
    {"nome": "Amapá", "sigla": "AP"},
    {"nome": "Amazonas", "sigla": "AM"},
    {"nome": "Bahia", "sigla": "BA"},
    {"nome": "Ceará", "sigla": "CE"},
    {"nome": "Distrito Federal", "sigla": "DF"},
    {"nome": "Espírito Santo", "sigla": "ES"},
    {"nome": "Goiás", "sigla": "GO"},
    {"nome": "Maranhão", "sigla": "MA"},
    {"nome": "Mato Grosso", "sigla": "MT"},
    {"nome": "Mato Grosso do Sul", "sigla": "MS"},
    {"nome": "Minas Gerais", "sigla": "MG"},
    {"nome": "Pará", "sigla": "PA"},
    {"nome": "Paraíba", "sigla": "PB"},
    {"nome": "Paraná", "sigla": "PR"},
    {"nome": "Pernambuco", "sigla": "PE"},
    {"nome": "Piauí", "sigla": "PI"},
    {"nome": "Rio de Janeiro", "sigla": "RJ"},
    {"nome": "Rio Grande do Norte", "sigla": "RN"},
    {"nome": "Rio Grande do Sul", "sigla": "RS"},
    {"nome": "Rondônia", "sigla": "RO"},
    {"nome": "Roraima", "sigla": "RR"},
    {"nome": "Santa Catarina", "sigla": "SC"},
    {"nome": "São Paulo", "sigla": "SP"},
    {"nome": "Sergipe", "sigla": "SE"},
    {"nome": "Tocantins", "sigla": "TO"},
]

nomes_estados = [estado["nome"] for estado in estados]

#Validando o campo CPF
def validar_cpf(cpf):
  
    cpf = cpf.replace(".", "").replace("-", "").strip()  # Remove pontos e traços

    # Verifica se o CPF tem exatamente 11 dígitos
    if len(cpf) != 11 or not cpf.isdigit():
        return False

    # Verifica se todos os números são iguais (e.g., 11111111111)
    if cpf == cpf[0] * 11:
        return False

    # Valida os dígitos verificadores
    for i in range(9, 11):
        soma = sum(int(cpf[j]) * ((i + 1) - j) for j in range(0, i))
        digito_verificador = ((soma * 10) % 11) % 10
        if int(cpf[i]) != digito_verificador:
            return False

    return True

def cadastrar_usuario():
    st.subheader("➕ Cadastro de novo usuário")

    # Coleta de informações do novo usuário
    cpf = st.text_input("CPF", max_chars=11, placeholder="Informe o CPF", label_visibility= "hidden")
    name = st.text_input("Nome", placeholder="Informe seu nome completo", label_visibility= "hidden")
    city = st.text_input("Cidade", placeholder="Informe sua cidade", label_visibility= "hidden")
    state = st.selectbox("Estado", nomes_estados, label_visibility= "hidden")
    password = st.text_input("Senha", type="password", placeholder="Senha", label_visibility= "hidden")
    confirm_password = st.text_input("Confirmar Senha", type="password", placeholder="Confirmar senha", label_visibility= "hidden")

    if st.button("Registrar"):
        if validar_cpf(cpf):
            if password != confirm_password:
                st.error("As senhas não coincidem!")
            elif cpf and password and city and state and name:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
                sucess = registrar_usuario(cpf,name, hashed_password, city, state)
                if sucess:
                    st.success(f"Usuário '{name}' cadastrado com sucesso!")
                else:
                    st.warning(f"CPF: {cpf} já está cadastrado com outro usuário!")
            else:
                st.error("Todos os campos são obrigatórios!")
        else:
            st.warning("CPF inválido!")
       
    
   #--------------------------------------------------------------------------------------------------------------------------------#
    
    # Mostrar mensagem de boas-vindas com o nome do usuário
    if 'cpf' in st.session_state and st.session_state['cpf']:
        st.sidebar.markdown(f"**{st.session_state['name']}**")
        st.sidebar.markdown(f"**{st.session_state['city']} - {st.session_state['state']}**")
   
    if st.sidebar.button("Sair"):
        # Redefine os estados e cookies do usuário
        st.session_state['authenticated'] = False
        st.session_state['cpf'] = None
        st.session_state['name'] = None
        st.session_state['city'] = None
        st.session_state['state'] = None

        delete_cookie("authenticated")
        delete_cookie("cpf")
        delete_cookie("name")
        delete_cookie("city")
        delete_cookie("state")
       
        st.rerun()
   

#---------------------------------------------------------------------------------------------------------------------------------#
# Função principal

def main():

    
    if st.session_state.get("authenticated", False):
        # Redireciona para a interface principal
        recognition_behavior()
    else:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
                logo1, logo2, logo3 = st.columns([5, 10, 5])
                with logo2:
                    
                    st.image(image_path, width=300)
                    
                # Variável de controle para a escolha da interface
                if "selected_option" not in st.session_state:
                    st.session_state["selected_option"] = "Login"

                # Renderizar o formulário baseado na escolha
                if st.session_state["selected_option"] == "Login":
                    login()
                elif st.session_state["selected_option"] == "Cadastrar":
                    cadastrar_usuario()

                # # Mostrar o rádio abaixo do formulário
                # radio1, radio2, radio3 = st.columns([3,2,3])
                # with radio2:
                #     st.radio(
                #         "Selecione uma opção:",
                #         ["Login", "Cadastrar"],
                #         index=["Login", "Cadastrar"].index(st.session_state["selected_option"]),
                #         key="selected_option",
                #         horizontal=True,
                #         label_visibility="hidden"
                #     )

                

if __name__ == "__main__":
    main()
