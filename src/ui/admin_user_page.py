import bcrypt
import streamlit as st

from control_database_postgres import list_all_users, registrar_usuario, reset_user_password


ESTADOS = [
    {"nome": "Informe o estado", "sigla": "BR"},
    {"nome": "Acre", "sigla": "AC"},
    {"nome": "Alagoas", "sigla": "AL"},
    {"nome": "Amapa", "sigla": "AP"},
    {"nome": "Amazonas", "sigla": "AM"},
    {"nome": "Bahia", "sigla": "BA"},
    {"nome": "Ceara", "sigla": "CE"},
    {"nome": "Distrito Federal", "sigla": "DF"},
    {"nome": "Espirito Santo", "sigla": "ES"},
    {"nome": "Goias", "sigla": "GO"},
    {"nome": "Maranhao", "sigla": "MA"},
    {"nome": "Mato Grosso", "sigla": "MT"},
    {"nome": "Mato Grosso do Sul", "sigla": "MS"},
    {"nome": "Minas Gerais", "sigla": "MG"},
    {"nome": "Para", "sigla": "PA"},
    {"nome": "Paraiba", "sigla": "PB"},
    {"nome": "Parana", "sigla": "PR"},
    {"nome": "Pernambuco", "sigla": "PE"},
    {"nome": "Piaui", "sigla": "PI"},
    {"nome": "Rio de Janeiro", "sigla": "RJ"},
    {"nome": "Rio Grande do Norte", "sigla": "RN"},
    {"nome": "Rio Grande do Sul", "sigla": "RS"},
    {"nome": "Rondonia", "sigla": "RO"},
    {"nome": "Roraima", "sigla": "RR"},
    {"nome": "Santa Catarina", "sigla": "SC"},
    {"nome": "Sao Paulo", "sigla": "SP"},
    {"nome": "Sergipe", "sigla": "SE"},
    {"nome": "Tocantins", "sigla": "TO"},
]


def validar_cpf(cpf: str) -> bool:
    cpf = (cpf or "").replace(".", "").replace("-", "").strip()

    if len(cpf) != 11 or not cpf.isdigit():
        return False

    if cpf == cpf[0] * 11:
        return False

    for i in range(9, 11):
        soma = sum(int(cpf[j]) * ((i + 1) - j) for j in range(0, i))
        digito_verificador = ((soma * 10) % 11) % 10
        if int(cpf[i]) != digito_verificador:
            return False

    return True


def _render_create_user_form():
    st.subheader("Novo usuario")
    with st.form("admin_create_user_form", clear_on_submit=True):
        cpf = st.text_input("CPF", max_chars=11, placeholder="Somente numeros")
        nome = st.text_input("Nome")
        cidade = st.text_input("Cidade")
        estado_options = [f"{e['sigla']} - {e['nome']}" for e in ESTADOS]
        estado_raw = st.selectbox("Estado", estado_options)
        role = st.selectbox("Perfil", ["professor", "admin"])
        email = st.text_input("Email")
        senha = st.text_input("Senha inicial", type="password")
        confirmar_senha = st.text_input("Confirmar senha inicial", type="password")
        submitted = st.form_submit_button("Criar usuario", type="primary", use_container_width=True)

    if not submitted:
        return

    estado = estado_raw.split(" - ")[0]
    cpf = (cpf or "").strip()
    nome = (nome or "").strip()
    cidade = (cidade or "").strip()
    email = (email or "").strip() or None

    if not validar_cpf(cpf):
        st.error("Informe um CPF valido.")
        return
    if not nome or not cidade or estado == "BR":
        st.error("Preencha nome, cidade e estado corretamente.")
        return
    if not senha:
        st.error("Informe uma senha inicial.")
        return
    if senha != confirmar_senha:
        st.error("As senhas nao coincidem.")
        return

    hashed_password = bcrypt.hashpw(senha.encode("utf-8"), bcrypt.gensalt()).decode()
    result = registrar_usuario(cpf, nome, hashed_password, cidade, estado, role=role, email=email)
    if result == "ok":
        st.success(f"Usuario '{nome}' criado com sucesso.")
    elif result == "cpf_exists":
        st.warning(f"O CPF {cpf} ja esta cadastrado.")
    else:
        st.error("Nao foi possivel criar o usuario.")


def _render_reset_password_form(users_df):
    st.subheader("Redefinir senha")
    if users_df.empty:
        st.info("Nenhum usuario cadastrado.")
        return

    user_options = {
        int(row["id"]): f"{row['nome']} ({row['cpf']}) - {row['role']}"
        for _, row in users_df.iterrows()
    }

    with st.form("admin_reset_password_form"):
        selected_user_id = st.selectbox(
            "Usuario",
            options=list(user_options.keys()),
            format_func=lambda user_id: user_options[user_id],
        )
        nova_senha = st.text_input("Nova senha", type="password")
        confirmar_nova_senha = st.text_input("Confirmar nova senha", type="password")
        submitted = st.form_submit_button("Salvar nova senha", type="primary", use_container_width=True)

    if not submitted:
        return

    if not nova_senha:
        st.error("Informe a nova senha.")
        return
    if nova_senha != confirmar_nova_senha:
        st.error("As senhas nao coincidem.")
        return

    hashed_password = bcrypt.hashpw(nova_senha.encode("utf-8"), bcrypt.gensalt()).decode()
    result = reset_user_password(selected_user_id, hashed_password)
    if result == "ok":
        st.success("Senha atualizada com sucesso.")
    elif result == "not_found":
        st.error("Usuario nao encontrado.")
    else:
        st.error("Nao foi possivel atualizar a senha.")


def render_admin_user_page(user_context: dict):
    if (user_context or {}).get("role") != "admin":
        st.error("Acesso restrito a administradores.")
        return

    st.title("Administracao de usuarios")
    users_df = list_all_users()

    st.subheader("Usuarios cadastrados")
    if users_df.empty:
        st.info("Nenhum usuario encontrado.")
    else:
        st.dataframe(
            users_df.rename(
                columns={
                    "id": "ID",
                    "nome": "Nome",
                    "cpf": "CPF",
                    "cidade": "Cidade",
                    "estado": "UF",
                    "email": "Email",
                    "role": "Perfil",
                    "ativo": "Ativo",
                    "created_at": "Criado em",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        _render_create_user_form()
    with col2:
        _render_reset_password_form(users_df)
