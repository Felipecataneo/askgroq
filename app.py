import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union

# CORREÇÃO CRÍTICA: Importar sse_client diretamente
from mcp.client.sse import sse_client # Agora esta importação deve funcionar
from mcp import ClientSession, types 
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# --- Configurações ---
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    st.error("Erro de configuração: A variável de ambiente MCP_SERVER_URL não está definida.")
    st.stop()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Erro de configuração: A variável de ambiente GROQ_API_KEY não está definida.")
    st.stop()

# --- Definições de Tipo (para ferramentas) ---
class ToolDefinition(TypedDict):
    type: str 
    function: Dict[str, Any]

# --- Cliente Principal para o Chatbot ---
class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str, groq_client: Groq):
        self.mcp_server_url = mcp_server_url
        self.groq_client = groq_client 

    # --- Funções Assíncronas ---

    # Método interno para conectar e manter as streams e a sessão MCP.
    # Esta é a parte complexa que sse_client normalmente abstrai via 'async with'.
    # Precisamos expor a sessão e as streams.
    async def _raw_connect_mcp_session(self) -> tuple[ClientSession, Any, Any]:
        """
        Conecta ao servidor MCP e retorna a sessão e as streams subjacentes.
        Isso contorna o uso de 'async with' para persistência.
        """
        st.info(f"Conectando ao servidor MCP em: {self.mcp_server_url}")
        try:
            # Obtém o transporte. sse_client é um gerador assíncrono.
            # O .__aenter__() é o que o 'async with' chamaria para iniciar o contexto.
            transport_context = sse_client(self.mcp_server_url)
            read_stream, write_stream = await transport_context.__aenter__() 
            
            # Cria a sessão MCP com as streams obtidas.
            session = ClientSession(read_stream, write_stream)
            await session.initialize() # Inicializa o protocolo MCP

            st.success("Conectado ao servidor MCP com sucesso!")
            
            # Retorna a sessão e o transport_context (para que __aexit__ não seja chamado)
            # Esta é a parte "hacky" para manter a conexão aberta.
            return session, transport_context, read_stream, write_stream
        except Exception as e:
            st.error(f"Erro ao conectar ao servidor MCP: {e}")
            raise 

    async def _load_mcp_tools_async(self, mcp_session: ClientSession) -> tuple[List[ToolDefinition], Dict[str, ClientSession]]:
        """Carrega e formata as ferramentas do servidor MCP."""
        try:
            response = await mcp_session.list_tools()
            tools = response.tools
            
            available_tools = []
            tool_to_session = {}
            for tool in tools:
                tool_to_session[tool.name] = mcp_session 
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            st.info(f"Ferramentas MCP carregadas: {[t['function']['name'] for t in available_tools]}")
            return available_tools, tool_to_session
        except Exception as e:
            st.error(f"Erro ao carregar ferramentas do servidor MCP: {e}")
            raise 

    async def process_query_async(self, query: str, mcp_session: ClientSession, available_tools: List[ToolDefinition], tool_to_session: Dict[str, ClientSession]):
        """Processa a query do usuário, interagindo com o Groq e ferramentas MCP."""
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
                    model="llama3-70b-8192",
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=2024
                )
            else:
                break 

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", page_icon="💡")
st.title("EIA Energy Data Chatbot (Powered by MCP & Groq)")
st.caption("Pergunte sobre dados de energia da EIA. Ex: 'Quais são as principais categorias de dados de energia na EIA?' ou 'Mostre-me os detalhes da rota 'electricity/retail-sales'.'")

# --- Inicialização Global ---
if "groq_client_instance" not in st.session_state:
    st.session_state.groq_client_instance = Groq(api_key=GROQ_API_KEY)

if "mcp_chatbot_logic_client" not in st.session_state:
    st.session_state.mcp_chatbot_logic_client = MCP_ChatBotClient(
        MCP_SERVER_URL, st.session_state.groq_client_instance
    )

# --- Conexão MCP Persistente usando st.cache_resource ---
# Esta função síncrona executa as corrotinas assíncronas usando asyncio.run()
# Retorna a sessão e as ferramentas para serem usadas no process_query.
@st.cache_resource(ttl=3600) 
def get_mcp_connection_resources(mcp_server_url: str) -> tuple[ClientSession, List[ToolDefinition], Dict[str, ClientSession]]:
    print(f"DEBUG: Tentando obter conexão MCP para {mcp_server_url}...")
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError: 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    temp_client_instance = MCP_ChatBotClient(mcp_server_url, st.session_state.groq_client_instance) 

    # CORREÇÃO: Chamar o método _raw_connect_mcp_session
    mcp_session, transport_context, read_stream, write_stream = loop.run_until_complete(temp_client_instance._raw_connect_mcp_session())
    available_tools, tool_to_session = loop.run_until_complete(temp_client_instance._load_mcp_tools_async(mcp_session))
    
    # Store transport_context to prevent __aexit__ from being called by GC
    # This is a hacky way to keep the connection open.
    st.session_state._mcp_transport_context = transport_context
    st.session_state._mcp_read_stream = read_stream
    st.session_state._mcp_write_stream = write_stream
    
    return mcp_session, available_tools, tool_to_session


# --- Inicialização da Conexão MCP ---
if "mcp_session" not in st.session_state: 
    st.session_state.connection_status = "pending"
    try:
        mcp_session, available_tools, tool_to_session = get_mcp_connection_resources(MCP_SERVER_URL)
        
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
elif st.session_state.connection_status == "connected" and not st.session_state.available_tools:
     st.warning("Conectado ao servidor MCP, mas nenhuma ferramenta carregada. Verifique os logs do servidor.")

# --- Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Entrada do Usuário e Processamento da Query ---
if st.session_state.get("connection_status") == "connected":
    if prompt := st.chat_input("Pergunte o que quiser..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(st.session_state.mcp_chatbot_logic_client.process_query_async(
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
        get_mcp_connection_resources.clear() 
        st.session_state.connection_status = "pending"
        st.session_state.messages = [] 
        st.rerun() 