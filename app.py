import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union

# CORREÇÃO: Importar sse_client diretamente.
from mcp.client.sse import sse_client
from mcp import ClientSession, types # ClientSession ainda é necessário
from groq import Groq
from dotenv import load_dotenv

# Importar streamlit_asyncio se estiver usando, senão remova.
# Se você removeu no requirements.txt, remova aqui também.
# Se o erro de Runtime do asyncio persistir, adicione novamente.
# import streamlit_asyncio as sa 

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    st.error("Erro de configuração: A variável de ambiente MCP_SERVER_URL não está definida.")
    st.stop()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Erro de configuração: A variável de ambiente GROQ_API_KEY não está definida.")
    st.stop()

class ToolDefinition(TypedDict):
    type: str 
    function: Dict[str, Any]

class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        # self.mcp_session não será mais um atributo direto aqui,
        # será gerenciado pelo st.cache_resource

    # Modifica para criar e retornar a sessão MCP e as ferramentas
    async def get_mcp_session_and_tools(self):
        """Conecta ao servidor MCP usando sse_client e retorna a sessão e ferramentas."""
        try:
            st.info(f"Conectando ao servidor MCP em: {self.mcp_server_url}")
            

            transport_instance = sse_client(self.mcp_server_url)
            read_stream, write_stream = await transport_instance._connect_async() 
            session = ClientSession(read_stream, write_stream)
            await session.initialize()

            st.success("Conectado ao servidor MCP com sucesso!")
            
            response = await session.list_tools()
            tools = response.tools
            
            available_tools = []
            tool_to_session = {}

            for tool in tools:
                tool_to_session[tool.name] = session
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            st.info(f"Ferramentas MCP carregadas: {[t['function']['name'] for t in available_tools]}")
            
            return session, available_tools, tool_to_session

        except Exception as e:
            st.error(f"Erro ao conectar ou inicializar o servidor MCP: {e}")
            raise # Levantar a exceção para ser pega pelo Streamlit

    async def process_query(self, query: str, mcp_session: ClientSession, available_tools: List[ToolDefinition], tool_to_session: Dict[str, ClientSession]):
        messages = [{'role':'user', 'content':query}]
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", 
            tools=available_tools,
            tool_choice="auto",
            max_tokens=2024
        )

        response_placeholder = st.empty() 
        
        while True: 
            response_message = chat_completion.choices[0].message
            
            if response_message.content:
                response_placeholder.markdown(response_message.content) 
            
            tool_calls = response_message.tool_calls
            
            if tool_calls:
                messages.append(response_message) 
                
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    st.info(f"Chamando ferramenta: {tool_name} com argumentos: {tool_args}")
                    
                    try:
                        result = await mcp_session.call_tool(tool_name, arguments=tool_args)
                        tool_output_content = json.dumps([item.dict() if hasattr(item, 'dict') else vars(item) for item in result.content]) 

                        st.success(f"Ferramenta {tool_name} executada com sucesso. Resultado:")
                        st.json(json.loads(tool_output_content)) 
                        
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "content": tool_output_content 
                            }
                        )
                    except Exception as e:
                        st.error(f"Erro ao executar a ferramenta {tool_name}: {e}")
                        messages.append(
                            {
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "content": json.dumps({"error": str(e), "tool_name": tool_name}) 
                            }
                        )
                
                chat_completion = self.groq_client.chat.completions.create(
                    messages=messages,
                    model="llama3-8b-8192",
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=2024
                )
            else:
                break 

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", page_icon="💡")
st.title("Chatbot (Powered by MCP & Groq)")
st.caption("Pergunte sobre dados de energia da EIA. Ex: 'Quais são as principais categorias de dados de energia na EIA?' ou 'Mostre-me os detalhes da rota 'electricity/retail-sales'.'")

if "groq_client_instance" not in st.session_state:
    st.session_state.groq_client_instance = Groq(api_key=GROQ_API_KEY)


# --- Conexão MCP via st.cache_resource ---
# Esta função memoizada conecta e retorna a sessão MCP e as ferramentas.
# Ela é executada apenas uma vez por worker/sessão do Streamlit (ou até o TTL expirar).
@st.cache_resource(ttl=3600) 
def get_mcp_connection(mcp_server_url: str):
    print(f"DEBUG: Tentando obter conexão MCP para {mcp_server_url}...")
    
    # Criar uma instância temporária do cliente para chamar o método assíncrono
    temp_client = MCP_ChatBotClient(mcp_server_url) 
    
    # Rodar a corrotina para conectar e pegar as ferramentas.
    # Esta é a parte sensível ao Runtime Error.
    # Se streamlit-asyncio não for usado, ou se ele não estiver "aplicado",
    # isso pode dar RuntimeError.
    
    # Tentativa de contornar o RuntimeError se estiver rodando um loop.
    # Mas no Streamlit Cloud, o Streamlit já gerencia um loop.
    # O ideal é usar o `streamlit-asyncio` ou um loop manual se não puder.
    
    # Vamos manter o asyncio.run, mas o erro do instalador é o principal.
    # Se o `ImportError` foi resolvido e este `RuntimeError` aparecer,
    # então a solução `streamlit-asyncio` é a correta.
    
    return asyncio.run(temp_client.get_mcp_session_and_tools())


# --- Inicialização da Conexão MCP ---
if "mcp_session" not in st.session_state: 
    st.session_state.connection_status = "pending"
    try:
        # AQUI CHAMAMOS A FUNÇÃO DE CONEXÃO
        # get_mcp_connection(MCP_SERVER_URL) retorna (session, tools, tool_map)
        mcp_session, available_tools, tool_to_session = get_mcp_connection(MCP_SERVER_URL)
        
        st.session_state.mcp_session = mcp_session
        st.session_state.available_tools = available_tools
        st.session_state.tool_to_session = tool_to_session
        st.session_state.connection_status = "connected"
    except Exception as e:
        st.session_state.connection_status = "failed"
        st.error(f"Falha na conexão inicial ao servidor MCP: {e}")
        st.stop()


# --- Exibição de Status de Conexão ---
if st.session_state.connection_status == "pending":
    st.info("Conectando ao servidor MCP...")
elif st.session_state.connection_status == "connected" and not st.session_state.chatbot_client.available_tools:
     st.warning("Conectado ao servidor MCP, mas nenhuma ferramenta carregada. Verifique os logs do servidor.")

# --- Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada do Usuário ---
if st.session_state.get("connection_status") == "connected":
    if prompt := st.chat_input("Pergunte o que quiser..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                # Use a instância de cliente Groq e as ferramentas da sessão Streamlit
                current_chatbot_client = MCP_ChatBotClient(MCP_SERVER_URL) # Instância temporária para o método
                current_chatbot_client.groq_client = st.session_state.groq_client_instance # Reatribui o cliente Groq

                # A chamada para process_query é assíncrona
                # Isso ainda usa asyncio.run, que pode ser o problema se o loop já estiver rodando.
                # A solução mais robusta é instalar e usar `streamlit-asyncio`.
                asyncio.run(current_chatbot_client.process_query(
                    prompt, 
                    st.session_state.mcp_session, 
                    st.session_state.available_tools, 
                    st.session_state.tool_to_session
                ))

else:
    st.warning("Chatbot não disponível. Verifique as configurações de conexão.")

# --- Botão de Reconexão (para depuração) ---
if st.button("Tentar reconectar ao servidor MCP"):
    if st.session_state.get("connection_status") != "connected":
        # Limpa o cache para forçar uma nova conexão
        get_mcp_connection.clear() # Limpa o cache da função memoizada
        st.session_state.connection_status = "pending"
        st.session_state.messages = [] # Limpa o histórico de chat
        st.rerun() 

# Lógica para fechar a sessão MCP: com st.cache_resource, ela é gerenciada pelo Streamlit.
# Não é necessário um cleanup manual com asyncio.run para o ClientSession se estiver dentro do cache.