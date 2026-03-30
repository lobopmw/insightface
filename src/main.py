
import streamlit as st
import bcrypt
import time
from sqlalchemy import text
from control_database_postgres import engine, get_user_by_cpf, get_user_context, registrar_usuario, user_table
from streamlit_cookies_controller import CookieController
from insightface_classroom import recognition_behavior
import os


#-----------------------------------------------------------------------------------------------------------------------------------#
# Configurações iniciais
page_icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/cam_IA.png"))
st.set_page_config(page_title="Monitoramento - SEDUC", page_icon=page_icon_path, layout="wide")

image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../images/classroom1.jpg"))
AUTH_COOKIE_NAME = "auth_user_cpf"
LEGACY_AUTH_QUERY_KEYS = ("authenticated", "cpf", "city", "state", "name", "role")
AUTH_RESTORE_BLOCK_KEY = "auth_restore_blocked"
AUTH_BOOTSTRAP_KEY = "auth_bootstrap_checked"
AUTH_BOOTSTRAP_STARTED_AT_KEY = "auth_bootstrap_started_at"
AUTH_BOOTSTRAP_GRACE_SECONDS = 0.6
cookie_controller = CookieController(key="auth_cookies")

# Criando/verificando a tabela de usuário uma vez por sessão
if "users_table_ready" not in st.session_state:
    user_table()
    st.session_state["users_table_ready"] = True


def apply_login_styles():
    st.markdown(
        """
        <style>
        div[data-testid="stForm"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background: linear-gradient(180deg, rgba(24, 27, 36, 0.96) 0%, rgba(17, 19, 27, 0.94) 100%);
            border-radius: 22px;
            padding: 1.2rem 1.1rem 1rem 1.1rem;
            box-shadow:
                0 24px 60px rgba(0, 0, 0, 0.42),
                0 8px 20px rgba(0, 0, 0, 0.22),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(14px);
        }

        div[data-testid="stTextInputRootElement"] {
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        div[data-testid="stTextInputRootElement"] input {
            font-size: 0.98rem;
            color: #f3f4f6;
        }

        div[data-testid="stTextInputRootElement"]:focus-within {
            border: 1px solid rgba(255, 255, 255, 0.14);
            box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def cookie_controller_is_ready() -> bool:
    try:
        cookies = cookie_controller.getAll()
        return isinstance(cookies, dict)
    except Exception:
        return False


def set_auth_cookie(cpf: str) -> None:
    if not cpf or not cookie_controller_is_ready():
        return
    try:
        cookie_controller.set(AUTH_COOKIE_NAME, cpf, path="/", same_site="strict")
    except Exception:
        pass


def get_auth_cookie():
    if not cookie_controller_is_ready():
        return None
    try:
        return cookie_controller.get(AUTH_COOKIE_NAME)
    except Exception:
        return None


def delete_auth_cookie() -> None:
    if not cookie_controller_is_ready():
        return
    try:
        if cookie_controller.get(AUTH_COOKIE_NAME) is not None:
            cookie_controller.remove(AUTH_COOKIE_NAME, path="/", same_site="strict")
    except Exception:
        pass


def clear_legacy_auth_query_params() -> bool:
    changed = False
    for key in LEGACY_AUTH_QUERY_KEYS:
        if key in st.query_params:
            del st.query_params[key]
            changed = True
    return changed


def reset_auth_session_state() -> None:
    st.session_state["authenticated"] = False
    st.session_state["cpf"] = None
    st.session_state["city"] = None
    st.session_state["state"] = None
    st.session_state["name"] = None
    st.session_state["role"] = None
    st.session_state.pop("user_context", None)


def request_logout() -> None:
    st.session_state[AUTH_RESTORE_BLOCK_KEY] = True
    st.session_state[AUTH_BOOTSTRAP_KEY] = True
    st.session_state[AUTH_BOOTSTRAP_STARTED_AT_KEY] = 0.0
    reset_auth_session_state()
    delete_auth_cookie()
    clear_legacy_auth_query_params()


def load_user_session_from_cpf(cpf: str) -> bool:
    if not cpf:
        return False

    cpf = str(cpf).strip()
    if not cpf:
        return False

    user = get_user_by_cpf(cpf)
    if not user:
        return False

    user_context = get_user_context(cpf)
    st.session_state["authenticated"] = True
    st.session_state["name"] = user["nome"]
    st.session_state["cpf"] = user["cpf"]
    st.session_state["city"] = user["cidade"]
    st.session_state["state"] = user["estado"]
    st.session_state["role"] = user["role"]
    st.session_state["user_context"] = user_context
    return True


def restore_auth_session_from_cookie() -> bool:
    if st.session_state.get("authenticated", False):
        st.session_state[AUTH_BOOTSTRAP_KEY] = True
        return True

    if st.session_state.get(AUTH_RESTORE_BLOCK_KEY):
        delete_auth_cookie()
        if get_auth_cookie():
            reset_auth_session_state()
            st.session_state[AUTH_BOOTSTRAP_KEY] = False
            return False
        st.session_state.pop(AUTH_RESTORE_BLOCK_KEY, None)
        st.session_state[AUTH_BOOTSTRAP_KEY] = True
        return False

    if not cookie_controller_is_ready():
        st.session_state[AUTH_BOOTSTRAP_KEY] = False
        return False

    stored_cpf = get_auth_cookie()
    st.session_state[AUTH_BOOTSTRAP_KEY] = True
    if not stored_cpf:
        return False

    if load_user_session_from_cpf(stored_cpf):
        return True

    delete_auth_cookie()
    reset_auth_session_state()
    return False


# Inicialização do estado da sessão
if "authenticated" not in st.session_state:
    reset_auth_session_state()

restore_auth_session_from_cookie()

legacy_auth_params_cleared = clear_legacy_auth_query_params()
if legacy_auth_params_cleared:
    st.rerun()


#----------------------------------------------------------------------------------------------------------------------------------------#
# Função Login

def login():
    
    # Coleta de informações de login
    colbutton1,colbutton2,colbutton3 = st.columns([1,3,1])
    with colbutton2:
        with st.form("login_form", clear_on_submit=False):
            cpf = st.text_input("CPF", placeholder="👤 CPF", max_chars=11, label_visibility= "hidden")
            password = st.text_input("Senha", type="password", placeholder="🔒 Senha", label_visibility= "hidden")
            submit_login = st.form_submit_button("**➡ Login**", use_container_width=True, type="primary")

        if submit_login:
            if validar_cpf(cpf):
                try:
                    user = get_user_by_cpf(cpf)
                    if user:
                        stored_nome = user["nome"]
                        stored_cpf = user["cpf"]
                        stored_password = user["password"]
                        stored_city = user["cidade"]
                        stored_state = user["estado"]
                        stored_role = user["role"]

                        try:
                            password_matches = bcrypt.checkpw(
                                password.encode('utf-8'),
                                stored_password.encode('utf-8')
                            )
                        except ValueError:
                            password_matches = stored_password == password
                            if password_matches:
                                new_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
                                with engine.connect() as conn:
                                    update_query = text("""
                                        UPDATE users
                                        SET password = :password
                                        WHERE cpf = :cpf
                                    """)
                                    conn.execute(update_query, {"password": new_hash, "cpf": stored_cpf})
                                    conn.commit()

                        if password_matches:
                            st.session_state['authenticated'] = True
                            st.session_state['name'] = stored_nome
                            st.session_state['cpf'] = stored_cpf
                            st.session_state['city'] = stored_city
                            st.session_state['state'] = stored_state
                            st.session_state['role'] = stored_role
                            st.session_state['user_context'] = get_user_context(stored_cpf)

                            set_auth_cookie(stored_cpf)
                            clear_legacy_auth_query_params()

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
    cpf = st.text_input("CPF", max_chars=11, placeholder="Informe o CPF", label_visibility="hidden")
    name = st.text_input("Nome", placeholder="Informe seu nome completo", label_visibility="hidden")
    city = st.text_input("Cidade", placeholder="Informe sua cidade", label_visibility="hidden")

    estado_options = [f"{e['sigla']} - {e['nome']}" if e['sigla'] != 'BR' else 'BR - Informe o estado' for e in estados]
    state_raw = st.selectbox("Estado", estado_options, label_visibility="hidden")
    state = state_raw.split(" - ")[0] if " - " in state_raw else state_raw

    password = st.text_input("Senha", type="password", placeholder="Senha", label_visibility="hidden")
    confirm_password = st.text_input("Confirmar Senha", type="password", placeholder="Confirmar senha", label_visibility="hidden")

    if st.button("Registrar"):
        if validar_cpf(cpf):
            if password != confirm_password:
                st.error("As senhas não coincidem!")
            elif not cpf or not name or not city or not state or state == 'BR':
                st.error("Todos os campos são obrigatórios e estado deve ser válido!")
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode()
                sucess = registrar_usuario(cpf, name, hashed_password, city, state)
                if sucess == "ok":
                    st.success(f"Usuário '{name}' cadastrado com sucesso!")
                elif sucess == "cpf_exists":
                    st.warning(f"CPF: {cpf} já está cadastrado com outro usuário!")
                else:
                    st.error("Erro ao cadastrar usuário. Verifique a conexão com o banco de dados e os logs.")
        else:
            st.warning("CPF inválido!")
       
    
   #--------------------------------------------------------------------------------------------------------------------------------#
    
    # Mostrar mensagem de boas-vindas com o nome do usuário
    if 'cpf' in st.session_state and st.session_state['cpf']:
        st.sidebar.markdown(f"**{st.session_state['name']}**")
        st.sidebar.markdown(f"**{st.session_state['city']} - {st.session_state['state']}**")
   
    if st.sidebar.button("Sair"):
        request_logout()
        st.rerun()
   

#---------------------------------------------------------------------------------------------------------------------------------#
# Função principal

def main():

    st.session_state.setdefault(AUTH_BOOTSTRAP_STARTED_AT_KEY, time.time())
    restore_auth_session_from_cookie()

    if (
        not st.session_state.get("authenticated", False)
        and not st.session_state.get(AUTH_RESTORE_BLOCK_KEY, False)
        and time.time() - st.session_state.get(AUTH_BOOTSTRAP_STARTED_AT_KEY, time.time())
        < AUTH_BOOTSTRAP_GRACE_SECONDS
    ):
        apply_login_styles()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            image_col1, image_col2, image_col3 = st.columns([1, 3, 1])
            with image_col2:
                st.image(image_path, use_container_width=True)
        return

    st.session_state[AUTH_BOOTSTRAP_STARTED_AT_KEY] = time.time()

    if st.session_state.get("authenticated", False):
        if "user_context" not in st.session_state and st.session_state.get("cpf"):
            st.session_state["user_context"] = get_user_context(st.session_state["cpf"])
        # Redireciona para a interface principal
        recognition_behavior()
    else:
        apply_login_styles()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
                image_col1, image_col2, image_col3 = st.columns([1, 3, 1])
                with image_col2:
                    st.image(image_path, use_container_width=True)

                login()

                

if __name__ == "__main__":
    main()
