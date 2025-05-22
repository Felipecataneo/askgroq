import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union
from mcp.client.sse import SseClientTransport
from mcp import ClientSession, types
from groq import Groq
from dotenv import load_dotenv

# Carrega variáveis de ambiente (para GROQ_API_KEY no ambiente local)
# EM STREAMLIT CLOUD, ESTE ARQUIVO .env NÃO SERÁ USADO.
# As variáveis serão lidas diretamente do ambiente de deploy.
load_dotenv()

# --- Configurações ---
# A URL do seu serviço MCP Server no Render
# Em Streamlit Cloud, essa URL será passada via Secrets.
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL") # REMOVI O VALOR DEFAULT LOCAL
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

# --- Inicialização de Cliente GROQ e Cliente MCP ---
class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.mcp_session: Optional[ClientSession] = None
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {} 
        self.is_connected = False # Nova flag para rastrear o estado da conexão

    async def connect_to_mcp_server(self):
        """Conecta ao servidor MCP via Streamable HTTP (SSE)."""
        if self.is_connected and self.mcp_session:
            return # Já conectado, evita reconexões desnecessárias

        try:
            st.info(f"Conectando ao servidor MCP em: {self.mcp_server_url}")
            transport = SseClientTransport(self.mcp_server_url)
            self.mcp_session = await transport.connect()
            
            st.success("Conectado ao servidor MCP com sucesso!")
            self.is_connected = True
            
            # Lista as ferramentas disponíveis
            response = await self.mcp_session.list_tools()
            tools = response.tools
            
            for tool in tools:
                self.tool_to_session[tool.name] = self.mcp_session 
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            st.info(f"Ferramentas MCP carregadas: {[t['function']['name'] for t in self.available_tools]}")

        except Exception as e:
            st.error(f"Erro ao conectar ou inicializar o servidor MCP: {e}")
            self.mcp_session = None 
            self.is_connected = False
            # Levantar a exceção para que o Streamlit saiba que falhou na conexão inicial
            raise

    async def process_query(self, query: str):
        # Garante que a conexão seja feita se ainda não estiver
        if not self.is_connected or not self.mcp_session:
            try:
                await self.connect_to_mcp_server()
            except Exception:
                st.error("Não foi possível estabelecer conexão com o servidor MCP. Por favor, verifique a URL e a disponibilidade do servidor.")
                return

        messages = [{'role':'user', 'content':query}]
        
        chat_completion = self.groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", 
            tools=self.available_tools,
            tool_choice="auto",
            max_tokens=2024
        )

        process_query_loop = True
        response_placeholder = st.empty() 
        
        while process_query_loop:
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
                        result = await self.mcp_session.call_tool(tool_name, arguments=tool_args)
                        # Adapta a saída do CallToolResult para o formato esperado pelo modelo (string JSON)
                        # mcp.types.TextContent tem um método .dict() para serializar em dict
                        # Outros tipos de content (ImageContent) também podem ter .dict() ou precisam ser tratados
                        tool_output_content = json.dumps([item.dict() if hasattr(item, 'dict') else vars(item) for item in result.content]) 

                        st.success(f"Ferramenta {tool_name} executada com sucesso. Resultado:")
                        st.json(json.loads(tool_output_content)) # Mostra o resultado JSON no Streamlit
                        
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
                    model="Llama3-70B-8192",
                    tools=self.available_tools,
                    tool_choice="auto",
                    max_tokens=2024
                )
            else:
                process_query_loop = False

    async def cleanup(self):
        if self.mcp_session:
            await self.mcp_session.close()

# --- Streamlit UI ---
st.set_page_config(page_title="Chatbot", page_icon="💡")
st.title("Chatbot (Powered by MCP & Groq)")
st.caption("Pergunte sobre dados de energia da EIA. Ex: 'Quais são as principais categorias de dados de energia na EIA?' ou 'Mostre-me os detalhes da rota 'electricity/retail-sales'.'")

# Inicializa o cliente chatbot e tenta conectar se não estiver na sessão
if "chatbot_client" not in st.session_state:
    st.session_state.chatbot_client = MCP_ChatBotClient(MCP_SERVER_URL)
    # A conexão inicial é feita aqui e armazenada na sessão.
    # Usamos st.session_state.connection_status para evitar tentativas repetidas de conexão na inicialização.
    st.session_state.connection_status = "pending"
    try:
        asyncio.run(st.session_state.chatbot_client.connect_to_mcp_server())
        st.session_state.connection_status = "connected"
    except Exception as e:
        st.session_state.connection_status = "failed"
        st.error(f"Não foi possível iniciar o chatbot devido a um erro de conexão inicial: {e}")
        st.stop() # Parar o Streamlit se a conexão inicial falhar

# Exibir mensagem de conexão apenas se estiver pendente ou conectado
if st.session_state.connection_status == "pending":
    st.info("Conectando ao servidor MCP...")
elif st.session_state.connection_status == "connected" and not st.session_state.chatbot_client.available_tools:
     st.warning("Conectado ao servidor MCP, mas nenhuma ferramenta carregada. Verifique os logs do servidor.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if st.session_state.get("connection_status") == "connected":
    if prompt := st.chat_input("Pergunte o que quiser..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processando..."):
                # Roda o processamento da query de forma assíncrona
                asyncio.run(st.session_state.chatbot_client.process_query(prompt))
                # Note: O histórico do chat aqui está sendo populado incrementalmente.
                # Para uma experiência perfeita, você pode precisar de uma lógica para
                # consolidar as saídas intermediárias (chamadas de ferramenta)
                # e adicionar apenas a resposta final do LLM ao `st.session_state.messages`.
                # Isso exigiria mais refatoração do `process_query` para retornar o texto final.
else:
    st.warning("Chatbot não disponível. Verifique as configurações de conexão.")

# Adicionar um botão de reconexão para depuração
if st.button("Tentar reconectar ao servidor MCP"):
    if st.session_state.get("connection_status") != "connected":
        st.session_state.connection_status = "pending"
        try:
            asyncio.run(st.session_state.chatbot_client.connect_to_mcp_server())
            st.session_state.connection_status = "connected"
            st.rerun() # Força o Streamlit a redesenhar a página
        except Exception:
            st.session_state.connection_status = "failed"
            st.error("Falha ao reconectar. Verifique a URL do servidor e os logs.")

# Lógica para fechar a sessão MCP quando o aplicativo Streamlit é fechado (ou reiniciado)
# Isso é um pouco complicado com Streamlit, pois ele não tem um lifecycle hook 'on_exit' confiável.
# Para evitar vazamento de recursos, as sessões TCP persistentes do SSEClientTransport
# idealmente deveriam ser gerenciadas por um AsyncExitStack ou fechadas explicitamente.
# No contexto do Streamlit Cloud, onde os workers são reiniciados periodicamente,
# as conexões antigas acabam sendo limpas, mas é bom ter uma tentativa de limpeza.