import streamlit as st
import os
from groq import Groq

# --- Configuração Inicial ---
st.set_page_config(page_title="Gerador de Conteúdo IA", layout="centered")

# Carregar a chave da API Groq
try:
    # Tenta carregar da variável de ambiente (para execução local)
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception: # Se falhar, tenta carregar dos secrets (para Streamlit Cloud)
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except KeyError: # Se a chave não for encontrada em nenhum lugar
        st.error("Chave de API GROQ_API_KEY não configurada. Verifique suas variáveis de ambiente ou secrets do Streamlit.")
        st.stop() # Para a execução do app

# --- Interface do Usuário Minimalista ---
st.markdown("""
<style>
    /* Remove padding excessivo do topo do container principal do Streamlit */
    .main > div {
        padding-top: 1rem;
    }

    /* Container das colunas (stHorizontalBlock) */
    div[data-testid="stHorizontalBlock"] {
        align-items: center; /* Alinha verticalmente os itens (colunas) ao centro */
        /* gap: 0.5rem; Adicionado via st.columns(gap="small") */
    }

    /* Campo de texto (st.text_input) */
    /* Streamlit text_input por padrão tem altura de 2.5rem (40px se 1rem=16px) */
    /* Não precisamos de muito estilo customizado para ele aqui se o label_visibility="collapsed" for usado */

    /* Coluna do Botão (o segundo filho do stHorizontalBlock) */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) {
        display: flex; /* Permite alinhar o conteúdo da coluna */
        align-items: center; /* Alinha o wrapper do botão (.stButton) verticalmente ao centro da coluna */
        justify-content: center; /* Centraliza o wrapper do botão horizontalmente na coluna */
        height: 2.5rem; /* Força a altura da coluna do botão a ser igual à do input */
    }

    /* Wrapper do Botão (stButton) e o Botão em si */
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button {
        height: 2.5rem; /* Altura do botão para corresponder ao input (40px) */
        width: 2.5rem;  /* Largura do botão para torná-lo quadrado (para o ícone) */
        min-width: 2.5rem; /* Garante que a largura não encolha */
        border: 1px solid #cccccc; /* Borda sutil, similar ao input */
        background-color: #f8f9fa; /* Cor de fundo suave */
        border-radius: 8px; /* Bordas arredondadas */
        display: flex; /* Para centralizar o ícone dentro do botão */
        align-items: center;
        justify-content: center;
        padding: 0; /* Remove padding padrão do botão, pois controlamos com altura/largura */
        font-size: 1.2rem; /* Tamanho do "ícone" (caractere) */
        color: #333; /* Cor do ícone */
        line-height: 1; /* Evita que a altura da linha adicione espaço extra */
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button:hover {
        background-color: #e9ecef; /* Efeito hover */
    }
    div[data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton button:active {
        background-color: #d3d9df; /* Efeito ao clicar */
    }

    /* Área de Resposta */
    .response-area {
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f9fa;
        min-height: 350px;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 1.5rem; /* Espaço acima da área de resposta */
    }
</style>
""", unsafe_allow_html=True)

# --- Layout com Colunas para Título e Botão ---
col1, col2 = st.columns([0.85, 0.15], gap="small") # 85% para input, 15% para botão, com um pequeno espaço

with col1:
    question = st.text_input(
        "Título/Pergunta:",  # Este rótulo será ocultado
        placeholder="Digite sua pergunta aqui...",
        label_visibility="collapsed" # Oculta o rótulo acima do campo
    )

with col2:
    # Usar uma chave única para o botão é uma boa prática
    submit_button = st.button("➤", help="Gerar resposta", key="submit_button_groq")

# Placeholder para a resposta
response_placeholder = st.container()

# --- Lógica da Aplicação ---

# Gerenciar o estado inicial da área de resposta
if 'user_interacted' not in st.session_state:
    st.session_state.user_interacted = False # Flag para saber se o usuário já clicou no botão

if not st.session_state.user_interacted:
    with response_placeholder:
        st.markdown("""
        <div class="response-area">
            <i>A resposta da IA aparecerá aqui...</i>
        </div>
        """, unsafe_allow_html=True)

if submit_button:
    st.session_state.user_interacted = True # Marcar que o usuário interagiu
    response_placeholder.empty() # Limpar conteúdo anterior (seja aviso, erro ou resultado antigo)

    if not question:
        with response_placeholder:
            st.warning("Por favor, digite uma pergunta.")
    else:
        with response_placeholder: # Usar o mesmo placeholder para spinner, erro ou resultado
            with st.spinner("Consultando a IA..."):
                try:
                    system_prompt = (
                        "Você é um assistente de IA da equipe do PEP-PERF. Explique conceitos de forma simples e direta para um público leigo. "
                        "Evite jargões ou explique-os. Use linguagem clara e concisa. "
                        "Formate sua resposta de maneira adequada para leitura, usando parágrafos ou listas se necessário."
                    )
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": question}
                        ],
                        model="llama3-8b-8192" # Modelo mais rápido para interações, ou o 70b
                    )
                    answer = chat_completion.choices[0].message.content

                    # Exibe a resposta na área designada
                    st.markdown(f"""
                    <div class="response-area">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erro ao contatar a IA: {e}")
