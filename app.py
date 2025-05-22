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
        error_msg = f"Erro ao conectar ao servidor MCP: {type(e).__name__}: {e}"
        logger.error(error_msg)
        
        # Tratar ExceptionGroup/TaskGroup para Python 3.11+
        if hasattr(e, 'exceptions') and e.exceptions:
            logger.error("Exceções subjacentes:")
            for i, sub_e in enumerate(e.exceptions):
                logger.error(f"  Sub-exceção {i+1}: {type(sub_e).__name__}: {sub_e}")
        
        # Sugestões baseadas no tipo de erro
        if "connection" in str(e).lower() or "refused" in str(e).lower():
            logger.error("💡 Sugestão: Verifique se o servidor MCP está rodando e acessível")
        elif "timeout" in str(e).lower():
            logger.error("💡 Sugestão: O servidor pode estar sobrecarregado ou lento")
        
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
# No arquivo do chatbot (onde você tem process_query_with_tools)

async def process_query_with_tools(query: str, groq_client: Groq, sse_url: str):
    """
    Processa uma query usando o padrão correto de conexão MCP.
    Esta função mantém a conexão ativa durante todo o processamento.
    """
    async with sse_client(sse_url) as (in_stream, out_stream):
        async with ClientSession(in_stream, out_stream) as session:
            await session.initialize()
            
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

            # --- PROMPT DE SISTEMA REFORÇADO ---
            system_prompt = """
            Você é um assistente especializado em dados de energia da U.S. Energy Information Administration (EIA), acessando dados através de uma API v2.
            Seu objetivo é responder às perguntas dos usuários usando as seguintes ferramentas: `list_eia_v2_routes`, `get_eia_v2_route_data`, e `get_eia_v2_series_id_data`.

            **REGRAS CRÍTICAS PARA USO DAS FERRAMENTAS:**

            1.  **FLUXO DE DESCOBERTA OBRIGATÓRIO (para perguntas abertas sobre dados):**
                *   **NÃO assuma rotas ou IDs.** Para perguntas como "Qual a produção de X em Y?" ou "Dados sobre Z", você DEVE seguir este fluxo:
                *   **Passo 1:** Comece com `list_eia_v2_routes()` (sem argumentos) para ver as categorias de nível superior.
                *   **Passo 2:** Analise a saída. Se uma categoria parecer relevante (ex: "petroleum"), chame `list_eia_v2_routes(segment_path="nome_da_categoria")` para ver sub-rotas e metadados.
                *   **Passo 3:** Continue chamando `list_eia_v2_routes` com `segment_path` cada vez mais específico até encontrar:
                    *   A rota base que contém os dados (ex: "petroleum/supply/monthly").
                    *   Os `ID do Facet` relevantes nos metadados (ex: `countryRegionId`, `productId`).
                    *   Os `ID da Coluna` relevantes para os dados solicitados (ex: `value` para produção).
                    *   O `ID (para query)` da frequência desejada (ex: `A` para anual, `M` para mensal).
                *   **Passo 4 (Se houver Facets):** Para cada `ID do Facet` que você precisa filtrar (ex: para encontrar "Brasil"), chame `list_eia_v2_routes(segment_path="<rota_base_encontrada>/facet/<ID_do_Facet_desejado>")`. Analise a saída para encontrar o `ID (valor do facet)` correspondente (ex: "BRA" para Brasil).
                *   **Passo 5:** SOMENTE APÓS completar os passos acima, use `get_eia_v2_route_data` com:
                    *   `route_path_with_data_segment`: A rota base encontrada + "/data/" (ex: "petroleum/supply/monthly/data/").
                    *   `data_elements`: Lista dos `ID da Coluna` encontrados.
                    *   `facets`: Dicionário com `{ID_do_Facet: ID_valor_do_facet}`.
                    *   `frequency`: O `ID (para query)` da frequência.
                    *   `start_period`, `end_period` conforme necessário (ex: para "este ano").

            2.  **USO DE `get_eia_v2_series_id_data`:**
                *   Use esta ferramenta **SOMENTE E EXCLUSIVAMENTE** se o usuário fornecer explicitamente um Series ID completo e formatado como um ID da APIv1 da EIA (ex: "ELEC.SALES.CO-RES.A").
                *   **NÃO invente, NÃO adivinhe, NÃO construa Series IDs.** Se o usuário não fornecer um Series ID da APIv1, IGNORE esta ferramenta e siga o FLUXO DE DESCOBERTA OBRIGATÓRIO.

            3.  **INTERPRETAÇÃO DA SAÍDA DE `list_eia_v2_routes`:**
                *   Preste muita atenção aos "ID da Coluna", "ID do Facet", "ID (valor do facet)", e "ID (para query)" da frequência. São esses IDs que você deve usar nos parâmetros das outras ferramentas. Não use os "Nomes" descritivos diretamente como IDs.

            4.  **PERSISTÊNCIA E MÚLTIPLAS CHAMADAS:**
                *   É esperado que você faça MÚLTIPLAS chamadas a `list_eia_v2_routes` para encontrar os dados corretos. Não tente adivinhar após a primeira chamada. Seja metódico.

            5.  **ANO CORRENTE:**
                *   Para "este ano" ou "ano corrente", use o ano atual. Se os dados do ano atual não estiverem completos, você pode mencionar isso e fornecer os dados do último ano completo disponível, se apropriado, ou perguntar ao usuário se ele deseja os dados mais recentes, mesmo que parciais.

            Pense passo a passo e explique seu raciocínio antes de chamar uma ferramenta, especialmente ao seguir o fluxo de descoberta.
            Se você não conseguir encontrar os dados após seguir o fluxo, informe ao usuário que não foi possível localizar os dados específicos com as ferramentas disponíveis.
            """

            messages = [
                {'role': 'system', 'content': system_prompt}, # Adiciona o prompt do sistema
                {'role': 'user', 'content': query}
            ]
            
            response_placeholder = st.empty()
            
            while True:
                # Chamada para o Groq
                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    model="llama3-70b-8192", 
                    tools=available_tools,
                    tool_choice="auto", # ou "required" se você quiser forçar uma ferramenta após a primeira resposta com tools
                    max_tokens=4096 # Aumentar um pouco para respostas mais longas/detalhadas e tool calls
                )
                
                response_message = chat_completion.choices[0].message
                
                # Debug: Mostrar a mensagem completa da resposta do LLM
                # st.write("Resposta do LLM:")
                # st.json(response_message.dict()) # .dict() é útil para Pydantic models
                logger.info(f"LLM Response: {response_message}")

                if response_message.content:
                    response_placeholder.markdown(response_message.content)
                
                tool_calls = response_message.tool_calls
                
                if tool_calls:
                    # Adicionar a resposta do assistente (que contém a decisão de usar a ferramenta) às mensagens
                    messages.append(response_message) 
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            tool_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as e:
                            st.error(f"Erro ao decodificar argumentos da ferramenta {tool_name}: {tool_call.function.arguments}. Erro: {e}")
                            messages.append({
                                "tool_call_id": tool_call.id,
                                "role": "tool",
                                "name": tool_name,
                                "content": json.dumps({"error": f"Argumentos JSON inválidos: {e}", "arguments_received": tool_call.function.arguments})
                            })
                            continue # Pula para o próximo tool_call ou iteração

                        st.info(f"Executando ferramenta: {tool_name}")
                        st.json(tool_args) # Mostrar argumentos
                        
                        tool_output = await execute_tool_call(session, tool_name, tool_args)
                        
                        try:
                            # Tentar parsear o output como JSON para melhor visualização, se for o caso
                            tool_output_json = json.loads(tool_output)
                            st.success(f"Resultado da ferramenta {tool_name}:")
                            st.json(tool_output_json)
                        except json.JSONDecodeError:
                            st.success(f"Resultado da ferramenta {tool_name} (texto):")
                            st.text(tool_output)

                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name, # Adicionar o nome da ferramenta aqui é bom para o LLM
                            "content": tool_output,
                        })
                    # Loop de volta para o LLM processar os resultados da ferramenta
                else:
                    # Se não houver tool_calls, a conversa para esta rodada terminou
                    # Não precisa adicionar response_message novamente se já tem conteúdo e não há tools
                    break 
            
            return response_message.content # Retorna o conteúdo final do assistente

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
def test_mcp_connection(server_url: str) -> tuple[bool, str]:
    """Testa a conexão com o servidor MCP."""
    async def _test():
        try:
            async with sse_client(server_url) as (in_stream, out_stream):
                async with ClientSession(in_stream, out_stream) as session:
                    await session.initialize()
                    return True, "Conexão bem-sucedida"
        except Exception as e:
            error_msg = f"Erro de conexão: {type(e).__name__}: {str(e)}"
            
            # Tratar TaskGroup/ExceptionGroup específicamente
            if hasattr(e, 'exceptions') and e.exceptions:
                error_details = []
                for i, sub_e in enumerate(e.exceptions):
                    error_details.append(f"Sub-erro {i+1}: {type(sub_e).__name__}: {str(sub_e)}")
                error_msg += f"\nDetalhes: {'; '.join(error_details)}"
            
            # Verificar se é erro de conexão de rede comum
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['connection', 'refused', 'timeout', 'unreachable']):
                error_msg += f"\n💡 Dica: Verifique se o servidor MCP está rodando em {server_url}"
            
            return False, error_msg
    
    try:
        return asyncio.run(_test())
    except Exception as e:
        return False, f"Erro crítico ao testar conexão: {type(e).__name__}: {str(e)}"

# --- Status de Conexão ---
connection_ok, connection_msg = test_mcp_connection(MCP_SERVER_URL)

if connection_ok:
    st.success("✅ Conectado ao servidor MCP")
else:
    st.error("❌ Falha na conexão com servidor MCP")
    with st.expander("Ver detalhes do erro", expanded=False):
        st.code(connection_msg, language="text")

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
    
    if st.button("🔧 Diagnóstico Detalhado"):
        st.write("Executando diagnóstico...")
        
        # Teste básico de URL
        if not MCP_SERVER_URL:
            st.error("URL do servidor MCP não configurada")
        else:
            st.info(f"URL configurada: {MCP_SERVER_URL}")
        
        # Teste de imports
        try:
            from mcp.client.sse import sse_client
            st.success("✅ Imports MCP OK")
        except ImportError as e:
            st.error(f"❌ Erro de import MCP: {e}")
        
        # Teste de conexão detalhado
        connection_ok, connection_msg = test_mcp_connection(MCP_SERVER_URL)
        if connection_ok:
            st.success("✅ Conexão MCP OK")
        else:
            st.error("❌ Conexão MCP falhou")
            st.code(connection_msg, language="text")
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"**Servidor MCP:** `{MCP_SERVER_URL}`")
    st.markdown(f"**Status:** {'🟢 Conectado' if connection_ok else '🔴 Desconectado'}")