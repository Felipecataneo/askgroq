import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union
import logging

# Importações corrigidas:
from mcp.client.sse import sse_client 
from mcp import ClientSession, types 
from groq import Groq
from dotenv import load_dotenv

# Reintroduzir nest_asyncio para garantir compatibilidade com asyncio.run() no Streamlit
import nest_asyncio 
nest_asyncio.apply() 

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Função Global para Conectar e Obter Ferramentas (Padrão Correto) ---
async def sse_get_tools_and_session(sse_url: str):
    """
    Conecta ao servidor MCP usando o padrão correto com async with
    e retorna session, ferramentas e mapeamento de ferramentas.
    """
    logger.info(f"Conectando ao servidor MCP em: {sse_url}")
    try:
        # PADRÃO CORRETO: Usar async with sse_client
        async with sse_client(sse_url) as (in_stream, out_stream):
            # Criar a sessão MCP sobre essas streams
            async with ClientSession(in_stream, out_stream) as session:
                logger.info(f"Conectado a {session.server_info.name} v{session.server_info.version}")
                
                # Inicializar a sessão
                await session.initialize()
                
                # Obter lista de ferramentas
                tools_result = await session.list_tools()
                
                # Formatear ferramentas para o Groq
                available_tools = []
                tool_to_session = {}
                
                for tool in tools_result.tools:
                    tool_to_session[tool.name] = session
                    available_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    })
                
                logger.info(f"Ferramentas carregadas: {[t['function']['name'] for t in available_tools]}")
                
                # IMPORTANTE: Retornar os dados, mas a sessão continuará ativa
                # dentro do contexto async with
                return session, available_tools, tool_to_session
                
    except Exception as e:
        logger.error(f"Erro ao conectar ao servidor MCP: {type(e).__name__}: {e}")
        # Tratar ExceptionGroup para Python 3.11+
        if hasattr(e, 'exceptions') and e.exceptions:
            logger.error("Exceções subjacentes do ExceptionGroup:")
            for sub_e in e.exceptions:
                logger.error(f"  - {type(sub_e).__name__}: {sub_e}")
        raise

# --- Função para Executar Tool Call ---
async def execute_tool_call(session: ClientSession, tool_name: str, tool_args: dict):
    """Executa uma chamada de ferramenta no servidor MCP."""
    try:
        result = await session.call_tool(tool_name, arguments=tool_args)
        
        # Formatar resultado
        tool_output_content = json.dumps([
            item.dict() if hasattr(item, 'dict') else vars(item) 
            for item in result.content
        ])
        
        return tool_output_content
        
    except Exception as e:
        logger.error(f"Erro ao executar ferramenta {tool_name}: {e}")
        return json.dumps({"error": str(e), "tool_name": tool_name})

# --- Função Principal de Processamento ---
async def process_query_with_tools(query: str, groq_client: Groq, sse_url: str):
    """
    Processa uma query usando o padrão correto de conexão MCP.
    Esta função mantém a conexão ativa durante todo o processamento.
    """
    # Conectar e obter ferramentas usando o padrão correto
    async with sse_client(sse_url) as (in_stream, out_stream):
        async with ClientSession(in_stream, out_stream) as session:
            await session.initialize()
            
            # Obter ferramentas
            tools_result = await session.list_tools()
            available_tools = []
            
            for tool in tools_result.tools:
                available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            st.info(f"Ferramentas disponíveis: {[t['function']['name'] for t in available_tools]}")
            
            # Iniciar conversa
            messages = [{'role': 'user', 'content': query}]
            
            # Placeholder para resposta
            response_placeholder = st.empty()
            
            # Loop de conversação
            while True:
                # Chamada para o Groq
                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    model="llama3-8b-8192", 
                    tools=available_tools,
                    tool_choice="auto",
                    max_tokens=2024
                )
                
                response_message = chat_completion.choices[0].message
                
                # Mostrar conteúdo da resposta
                if response_message.content:
                    response_placeholder.markdown(response_message.content)
                
                # Verificar se há chamadas de ferramentas
                tool_calls = response_message.tool_calls
                
                if tool_calls:
                    messages.append(response_message)
                    
                    # Executar cada ferramenta
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        st.info(f"Executando ferramenta: {tool_name}")
                        st.json(tool_args)
                        
                        # Executar ferramenta usando a sessão ativa
                        tool_output = await execute_tool_call(session, tool_name, tool_args)
                        
                        # Mostrar resultado
                        try:
                            result_data = json.loads(tool_output)
                            st.success(f"Ferramenta {tool_name} executada com sucesso:")
                            st.json(result_data)
                        except json.JSONDecodeError:
                            st.success(f"Resultado da ferramenta {tool_name}:")
                            st.text(tool_output)
                        
                        # Adicionar resultado às mensagens
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "content": tool_output
                        })
                    
                    # Continuar loop para próxima resposta
                    continue
                else:
                    # Sem mais ferramentas, finalizar
                    break
            
            return response_message.content

# --- Cliente Principal Simplificado ---
class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str, groq_client: Groq):
        self.mcp_server_url = mcp_server_url
        self.groq_client = groq_client

    async def process_query(self, query: str):
        """Processa uma query usando conexão MCP adequada."""
        return await process_query_with_tools(query, self.groq_client, self.mcp_server_url)

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot MCP", page_icon="💡")
st.title("EIA Energy Data Chatbot (Powered by MCP & Groq)")
st.caption("Pergunte sobre dados de energia da EIA. Ex: 'Quais são as principais categorias de dados de energia na EIA?'")

# --- Inicialização ---
if "groq_client_instance" not in st.session_state:
    st.session_state.groq_client_instance = Groq(api_key=GROQ_API_KEY)

if "mcp_chatbot_client" not in st.session_state:
    st.session_state.mcp_chatbot_client = MCP_ChatBotClient(
        MCP_SERVER_URL, 
        st.session_state.groq_client_instance
    )

# --- Teste de Conexão ---
@st.cache_data(ttl=300)  # Cache por 5 minutos
def test_mcp_connection(server_url: str) -> bool:
    """Testa a conexão com o servidor MCP."""
    async def _test():
        try:
            async with sse_client(server_url) as (in_stream, out_stream):
                async with ClientSession(in_stream, out_stream) as session:
                    await session.initialize()
                    return True
        except Exception as e:
            st.error(f"Erro de conexão: {e}")
            return False
    
    return asyncio.run(_test())

# --- Status de Conexão ---
if test_mcp_connection(MCP_SERVER_URL):
    st.success("✅ Conectado ao servidor MCP")
    connection_ok = True
else:
    st.error("❌ Falha na conexão com servidor MCP")
    connection_ok = False

# --- Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Interface de Chat ---
if connection_ok:
    if prompt := st.chat_input("Digite sua pergunta..."):
        # Adicionar mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Processar resposta
        with st.chat_message("assistant"):
            with st.spinner("Processando sua pergunta..."):
                try:
                    # Usar a nova função de processamento
                    response = asyncio.run(
                        st.session_state.mcp_chatbot_client.process_query(prompt)
                    )
                    
                    if response:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                except Exception as e:
                    error_msg = f"Erro ao processar consulta: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
else:
    st.warning("⚠️ Chatbot indisponível devido a problemas de conexão.")

# --- Controles de Depuração ---
with st.sidebar:
    st.header("Configurações")
    
    if st.button("🔄 Testar Conexão"):
        test_mcp_connection.clear()  # Limpar cache
        st.rerun()
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Servidor MCP:** `{MCP_SERVER_URL}`")
    st.markdown(f"**Status:** {'🟢 Conectado' if connection_ok else '🔴 Desconectado'}")