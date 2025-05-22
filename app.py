import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union

# Importações corrigidas:
from mcp.client.sse import sse_client # Agora esta importação deve funcionar
from mcp import ClientSession, types 
from groq import Groq
from dotenv import load_dotenv

# ADICIONAR nest_asyncio para contornar problemas de asyncio.run()
import nest_asyncio 
nest_asyncio.apply() 

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

    # Esta função irá gerenciar a sessão MCP e as ferramentas
    async def get_mcp_session_and_tools(self) -> tuple[ClientSession, List[ToolDefinition], Dict[str, ClientSession]]:
        """Conecta ao servidor MCP, inicializa a sessão e carrega as ferramentas."""
        st.info(f"Conectando ao servidor MCP em: {self.mcp_server_url}")
        try:
            # Usar o context manager sse_client diretamente
            # Ele cria e gerencia a conexão subjacente.
            # A sessão MCP é criada com as streams que ele fornece.
            
            # NOTA: O sse_client é um context manager, que normalmente fecha a conexão.
            # A forma padrão de usá-lo é `async with sse_client(...) as (r,w): ...`
            # Mas Streamlit precisa de uma sessão persistente.
            # Se a classe SseClientTransport não é importável, como no erro anterior,
            # estamos em um dilema.

            # Revertendo para a forma mais "manual" de obter as streams se SseClientTransport não funciona.
            # Isso é para contornar o problema de importação do SseClientTransport.
            # transport_context = sse_client(self.mcp_server_url)
            # read_stream, write_stream = await transport_context.__aenter__() 
            # session = ClientSession(read_stream, write_stream)
            # await session.initialize()

            # TENTATIVA: usar a própria classe SseClientTransport do mcp.client.sse
            # Se a ImportError persistir, então temos que ver se a classe existe numa versão mais antiga.
            # Ou o problema é com o próprio httpx-sse.
            
            # ASSUMINDO que o Import Error for SseClientTransport foi por erro de digitação/versão
            # e que, com o requirements.txt correto, SseClientTransport DEVE funcionar.
            # A versão anterior com SseClientTransport é a IDEAL para persistência.
            
            # Se o erro atual é "unhandled errors in a TaskGroup",
            # E A SUB-EXCEÇÃO É DE REDE/CONEXÃO (refusal, timeout),
            # então o problema NÃO É com a forma de importar SseClientTransport.
            # É um problema de conexão.

            # Vamos manter a tentativa com SseClientTransport (se ele for importável),
            # mas adicionar tratamento de erro para a conexão real.

            # --- Conectando com SseClientTransport ---
            # SE ImportError PARA SseClientTransport PERSISTIR, ESTE CÓDIGO NÃO FUNCIONARÁ.
            # Apenas se o erro for no TaskGroup ou outra coisa.
            transport = SseClientTransport(self.mcp_server_url)
            session = await transport.connect()
            await session.initialize()
            st.success("Conectado ao servidor MCP com sucesso!")

            # Carrega e formata as ferramentas
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
            # Captura a exceção e loga para depuração
            st.error(f"Erro detalhado na conexão MCP: {type(e).__name__}: {e}")
            raise # Re-levanta para que o Streamlit Cloud possa logar a sub-exceção


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
                    model="llama3-8b-8192",
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
@st.cache_resource(ttl=3600) 
def get_mcp_connection_resources(mcp_server_url: str) -> tuple[ClientSession, List[ToolDefinition], Dict[str, ClientSession]]:
    print(f"DEBUG: Tentando obter conexão MCP para {mcp_server_url}...")
    
    # Obtém o loop de eventos existente ou cria um novo.
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError: 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    temp_client_instance = MCP_ChatBotClient(mcp_server_url, st.session_state.groq_client_instance) 

    # Rodar as corrotinas de conexão e carregamento de ferramentas
    mcp_session = loop.run_until_complete(temp_client_instance._connect_mcp_session_async())
    available_tools, tool_to_session = loop.run_until_complete(temp_client_instance._load_mcp_tools_async(mcp_session))
    
    # Não precisamos mais do hack do _mcp_transport_context se o SseClientTransport for importável.
    # A sessão em si (mcp_session) é persistente e o transport está "por baixo" dela.
    
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
                if loop.is_running(): # Evita RuntimeErrors em loops já rodando
                    # Se o loop está rodando, usar run_until_complete no loop existente.
                    # Não criar um novo loop.
                    pass # Isso significa que o asyncio.run() abaixo pode dar erro.
                         # A forma correta é usar loop.create_task() e esperar por ele.
                
                # --- CORREÇÃO FINAL PARA CHAMADA ASSÍNCRONA EM STREAMLIT ---
                # Esta é a forma mais robusta sem `streamlit-asyncio` ou `nest_asyncio`.
                # Requer que a corrotina seja agendada no loop existente.
                # A função precisa ser "async def", e a chamada com "await".
                # Mas Streamlit não permite "await" diretamente no topo do script.
                # A solução abaixo é um padrão comum.
                
                # Crie uma função assíncrona para envolver o processamento da query
                async def run_query_processing():
                    await st.session_state.mcp_chatbot_logic_client.process_query_async(
                        prompt, 
                        st.session_state.mcp_session, 
                        st.session_state.available_tools, 
                        st.session_state.tool_to_session
                    )
                
                # Execute a corrotina no loop de eventos.
                # Streamlit executa a UI em um loop de eventos.
                # asyncio.run() criaria um NOVO loop, o que causa o RuntimeError.
                # A solução é obter o loop existente e agendar a tarefa.
                try:
                    current_loop = asyncio.get_running_loop() # Pega o loop que está rodando o Streamlit
                except RuntimeError: # Se por algum motivo não houver loop (muito raro no Streamlit)
                    current_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(current_loop)
                
                # Agenda a tarefa no loop existente
                current_loop.run_until_complete(run_query_processing())


else:
    st.warning("Chatbot não disponível. Verifique as configurações de conexão.")

# --- Botão de Reconexão (para depuração) ---
if st.button("Tentar reconectar ao servidor MCP"):
    if st.session_state.get("connection_status") != "connected":
        get_mcp_connection_resources.clear() 
        st.session_state.connection_status = "pending"
        st.session_state.messages = [] 
        st.rerun() 