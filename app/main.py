import streamlit as st
import os
from groq import Groq

# --- Configuração Inicial ---
st.set_page_config(page_title="Gerador de Conteúdo IA", layout="centered")

# Carregar a chave da API Groq
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    except KeyError:
        st.error("Chave de API GROQ_API_KEY não configurada.")
        st.stop()

# --- Interface do Usuário Minimalista ---

# CSS Customizado (ajustado para minimalismo)
st.markdown("""
<style>
    .main > div { /* Remove padding excessivo do container principal do Streamlit */
        padding-top: 1rem;
    }
    .title-input-container div[data-baseweb="input"] > div { /* Campo de input */
        border-radius: 8px;
        border: 1px solid #cccccc;
        font-size: 18px; /* Tamanho da fonte dentro do input */
        padding-left: 10px; /* Espaço interno */
    }
    .response-area {
        border: 1px solid #e9ecef; /* Borda mais suave */
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f9fa; /* Fundo suave */
        min-height: 350px;
        font-size: 16px;
        line-height: 1.6;
        margin-top: 1.5rem; /* Espaço acima da área de resposta */
    }
    /* Ajuste para o botão/ícone ao lado do input */
    div[data-testid="stHorizontalBlock"] {
        align-items: end; /* Alinha itens na base dentro da coluna */
    }
    /* Estilo para o botão como ícone (exemplo simples) */
    .stButton>button {
        border: none;
        background-color: transparent;
        padding: 0.5rem 0.5rem; /* Ajuste o padding para o tamanho do ícone */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton>button:hover {
        background-color: #f0f0f0; /* Um leve hover */
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

# --- Layout com Colunas para Título e Botão ---
col1, col2 = st.columns([0.85, 0.15]) # Proporção das colunas: 85% para input, 15% para botão

with col1:
    # Usamos um container com classe CSS para estilizar o input se necessário
    st.markdown('<div class="title-input-container">', unsafe_allow_html=True)
    question = st.text_input(
        "Título/Pergunta:",  # Rótulo simples, pode até ser removido se a placeholder for clara
        placeholder="Digite sua pergunta aqui...",
        label_visibility="collapsed" # Oculta o rótulo "Título/Pergunta:" se a placeholder for suficiente
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # O Streamlit não suporta ícones dentro de st.button nativamente de forma simples.
    # Usaremos um emoji ou um caractere Unicode como "ícone".
    # Para ícones reais, seria preciso HTML/JS mais complexo ou componentes customizados.
    # O botão será renderizado logo abaixo do input devido ao fluxo de colunas,
    # mas a estilização pode tentar aproximá-lo.
    # Para alinhar verticalmente o botão com o input, é um desafio com st.button padrão.
    # A CSS `align-items: end;` na `stHorizontalBlock` ajuda.
    submit_button = st.button("➤", help="Gerar resposta") # "➤" como ícone de "play" ou "enviar"

# Placeholder para a resposta (corpo do slide)
response_placeholder = st.container() # Usamos container para poder adicionar múltiplos elementos se necessário

# --- Lógica da Aplicação ---
if submit_button:
    if not question:
        with response_placeholder:
            st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("Consultando a IA..."):
            try:
                system_prompt = (
                    "Você é um assistente de IA. Explique conceitos de forma simples e direta para um público leigo. "
                    "Evite jargões ou explique-os. Use linguagem clara e concisa. "
                    "Formate sua resposta de maneira adequada para leitura, usando parágrafos ou listas se necessário."
                )
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    model="llama3-8b-8192" # Modelo mais rápido para interações rápidas, ou o 70b se preferir
                )
                answer = chat_completion.choices[0].message.content

                # Exibe a resposta na área designada
                with response_placeholder:
                    # Limpa o placeholder antes de adicionar novo conteúdo (caso haja aviso anterior)
                    response_placeholder.empty()
                    st.markdown(f"""
                    <div class="response-area">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                with response_placeholder:
                    response_placeholder.empty()
                    st.error(f"Erro ao contatar a IA: {e}")
else:
    # Mantém a área de resposta visível, mas vazia ou com uma mensagem inicial
    with response_placeholder:
        st.markdown(f"""
        <div class="response-area">
            <i>A resposta da IA aparecerá aqui...</i>
        </div>
        """, unsafe_allow_html=True)
