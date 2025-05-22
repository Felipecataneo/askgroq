import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
import logging
import traceback
import sys
from contextlib import asynccontextmanager
import threading
import time

# Importações corrigidas para usar o padrão genai.Client
from mcp.client.sse import sse_client
from mcp import ClientSession, types as mcp_types
from google import genai # Usar este, conforme seu exemplo
from google.genai import types # Usar este, conforme seu exemplo
from dotenv import load_dotenv

# Configuração especial do asyncio para Streamlit
import nest_asyncio
nest_asyncio.apply()

# Força o uso de SelectorEventLoop no Windows para evitar problemas de concorrência
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configurações ---
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL")
if not MCP_SERVER_URL:
    st.error("Erro de configuração: A variável de ambiente MCP_SERVER_URL não está definida.")
    st.stop()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Erro de configuração: A variável de ambiente GEMINI_API_KEY não está definida.")
    st.stop()

# --- Streamlit UI Setup (carrega primeiro) ---
st.set_page_config(page_title="Chatbot MCP", page_icon="💡", layout="wide")
st.title("🔌 EIA Energy Data Chatbot")
st.caption("💡 Powered by MCP & Gemini | Pergunte about dados de energia da EIA")

# --- Estado da Aplicação ---
if "initialization_complete" not in st.session_state:
    st.session_state.initialization_complete = False
if "connection_status" not in st.session_state:
    st.session_state.connection_status = {"status": "unknown", "message": "Não testado"}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "server_wake_attempted" not in st.session_state:
    st.session_state.server_wake_attempted = False

# --- Utilitários para Tratamento de Exceções ---
def extract_nested_exception_details(exception: Exception) -> Dict[str, Any]:
    """Extrai detalhes de exceções aninhadas, incluindo ExceptionGroup/TaskGroup."""
    details = {
        "type": type(exception).__name__,
        "message": str(exception),
        "nested_exceptions": []
    }
    
    # Verifica se é um ExceptionGroup (Python 3.11+) ou tem atributo exceptions
    if hasattr(exception, 'exceptions') and exception.exceptions:
        for i, sub_exception in enumerate(exception.exceptions):
            sub_details = extract_nested_exception_details(sub_exception)
            sub_details["index"] = i
            details["nested_exceptions"].append(sub_details)
    
    # Verifica se tem __cause__ ou __context__
    if exception.__cause__:
        details["cause"] = extract_nested_exception_details(exception.__cause__)
    if exception.__context__ and exception.__context__ != exception.__context__:
        details["context"] = extract_nested_exception_details(exception.__context__)
    
    return details

def format_exception_message(exception: Exception) -> str:
    """Formata uma mensagem de erro user-friendly a partir de uma exceção."""
    details = extract_nested_exception_details(exception)
    
    def _format_recursive(details_dict: Dict, level: int = 0) -> str:
        indent = "  " * level
        message = f"{indent}• {details_dict['type']}: {details_dict['message']}\n"
        
        for nested in details_dict.get("nested_exceptions", []):
            message += _format_recursive(nested, level + 1)
        
        if "cause" in details_dict:
            message += f"{indent}Causado por:\n"
            message += _format_recursive(details_dict["cause"], level + 1)
        
        return message
    
    return _format_recursive(details).strip()

async def wait_for_server_ready(server_url: str, max_wait_time: int = 120) -> bool:
    """
    Aguarda até que o servidor MCP esteja realmente pronto para aceitar conexões.
    """
    import httpx
    
    base_url = server_url.replace('/sse', '')
    start_time = time.time()
    
    logger.info(f"Aguardando servidor MCP ficar pronto: {base_url}")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        while (time.time() - start_time) < max_wait_time:
            try:
                # Tenta o endpoint SSE diretamente
                response = await client.get(server_url, timeout=10.0)
                if response.status_code == 200:
                    logger.info("Servidor MCP está respondendo corretamente")
                    return True
            except Exception as e:
                logger.debug(f"Servidor ainda não está pronto: {e}")
            
            await asyncio.sleep(3)  # Aguarda 3 segundos entre tentativas
    
    logger.warning(f"Servidor não ficou pronto em {max_wait_time}s")
    return False

# --- Função para "Acordar" o Servidor ---
async def wake_up_server(server_url: str) -> bool:
    """
    Tenta acordar um servidor que pode estar em cold start (como Render.com).
    Versão melhorada com mais tempo de espera.
    """
    import httpx
    
    base_url = server_url.replace('/sse', '')
    health_endpoints = [
        f"{base_url}/health",
        f"{base_url}/status", 
        f"{base_url}/",
        server_url  # Tenta o próprio endpoint SSE
    ]
    
    logger.info(f"Tentando acordar servidor: {base_url}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:  # Timeout maior
        for attempt in range(3):
            for endpoint in health_endpoints:
                try:
                    logger.info(f"Wake-up attempt {attempt + 1}: {endpoint}")
                    response = await client.get(endpoint)
                    if response.status_code < 500:
                        logger.info(f"Servidor respondeu: {endpoint} -> {response.status_code}")
                        # CORREÇÃO: Aguardar mais tempo para estabilização
                        await asyncio.sleep(8)  # Aumentado de 2s para 8s
                        
                        # CORREÇÃO: Verificar se servidor está realmente pronto
                        if await wait_for_server_ready(server_url, max_wait_time=60):
                            return True
                        else:
                            logger.warning("Servidor respondeu mas não está pronto para MCP")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Wake-up falhou para {endpoint}: {e}")
                    continue
            
            if attempt < 2:
                wait_time = (attempt + 1) * 10  # CORREÇÃO: 10s, 20s (mais tempo)
                logger.info(f"Esperando {wait_time}s antes da próxima tentativa...")
                await asyncio.sleep(wait_time)
    
    return False

# --- Context Manager para Sessões MCP com Wake-up ---
@asynccontextmanager
async def mcp_session(server_url: str, timeout: float = 45.0, wake_up: bool = True):
    """Context manager para gerenciar sessões MCP com cleanup adequado e wake-up opcional."""
    session = None
    streams = None
    try:
        # Tentar acordar o servidor se solicitado
        if wake_up and not st.session_state.server_wake_attempted:
            logger.info("Tentando acordar servidor MCP...")
            wake_success = await wake_up_server(server_url)
            st.session_state.server_wake_attempted = True
            if wake_success:
                logger.info("Servidor acordado com sucesso")
                # CORREÇÃO: Aguardar mais tempo após acordar o servidor
                await asyncio.sleep(5)
            else:
                logger.warning("Não foi possível confirmar se o servidor acordou")
        
        logger.info(f"Estabelecendo conexão MCP com {server_url}")
        
        # CORREÇÃO: Timeout mais longo para conexão inicial
        streams = sse_client(server_url)
        in_stream, out_stream = await asyncio.wait_for(
            streams.__aenter__(), 
            timeout=timeout
        )
        
        session = ClientSession(in_stream, out_stream)
        
        # CORREÇÃO: Timeout muito mais longo para inicialização e retry logic
        init_timeout = max(30.0, timeout / 2)  # Pelo menos 30s para inicialização
        max_init_retries = 2
        
        for attempt in range(max_init_retries + 1):
            try:
                logger.info(f"Tentativa {attempt + 1} de inicialização da sessão MCP")
                await asyncio.wait_for(session.initialize(), timeout=init_timeout)
                logger.info("Sessão MCP inicializada com sucesso")
                break
            except asyncio.TimeoutError:
                if attempt < max_init_retries:
                    logger.warning(f"Timeout na inicialização (tentativa {attempt + 1}). Tentando novamente...")
                    await asyncio.sleep(2)
                    continue
                else:
                    raise asyncio.TimeoutError(f"Timeout na inicialização da sessão MCP após {attempt + 1} tentativas")
        
        yield session
        
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout ao conectar com MCP: {e}")
        if "inicialização" in str(e):
            raise Exception(f"Timeout na inicialização da sessão MCP. O servidor está respondendo mas não consegue completar o handshake. Tente novamente.")
        else:
            raise Exception(f"Timeout na conexão MCP após {timeout}s. O servidor pode estar em cold start. Tente novamente em alguns momentos.")
    except Exception as e:
        logger.error(f"Erro na sessão MCP: {e}")
        raise
    finally:
        logger.info("Limpando recursos da sessão MCP")
        try:
            if session:
                # CORREÇÃO: Cleanup mais robusto
                if hasattr(session, 'close'):
                    await asyncio.wait_for(session.close(), timeout=5.0)
                elif hasattr(session, '_transport') and session._transport:
                    if hasattr(session._transport, 'close'):
                        await asyncio.wait_for(session._transport.close(), timeout=5.0)
        except Exception as cleanup_error:
            logger.warning(f"Erro durante cleanup da sessão: {cleanup_error}")
        
        try:
            if streams:
                await asyncio.wait_for(streams.__aexit__(None, None, None), timeout=5.0)
        except Exception as cleanup_error:
            logger.warning(f"Erro durante cleanup dos streams: {cleanup_error}")

# --- Função para Executar Tool Call com Timeout ---
async def execute_tool_call_safe(session: ClientSession, tool_name: str, tool_args: dict, timeout: float = 45.0) -> Union[Dict, List]:
    """
    Executa uma chamada de ferramenta no servidor MCP com timeout e tratamento robusto de erros.
    """
    try:
        logger.info(f"Executando ferramenta: {tool_name} com argumentos: {tool_args}")
        
        # Executa a chamada da ferramenta com timeout
        result = await asyncio.wait_for(
            session.call_tool(tool_name, arguments=tool_args),
            timeout=timeout
        )

        if hasattr(result, 'content') and result.content is not None:
            tool_output_content = []
            for item in result.content:
                if hasattr(item, 'model_dump'): # Pydantic v2
                    tool_output_content.append(item.model_dump())
                elif hasattr(item, 'dict'): # Pydantic v1
                    tool_output_content.append(item.dict())
                elif hasattr(item, '__dict__'): # Generic object
                    tool_output_content.append(vars(item))
                else:
                    tool_output_content.append(str(item))

            if len(tool_output_content) == 1 and isinstance(tool_output_content[0], dict):
                return tool_output_content[0]
            return tool_output_content if tool_output_content else {"result": "Nenhum conteúdo retornado", "tool_name": tool_name}
        else:
            return {"result": "Nenhum conteúdo retornado ou formato inesperado", "tool_name": tool_name}

    except asyncio.TimeoutError:
        error_msg = f"Timeout ao executar ferramenta {tool_name} (>{timeout}s)"
        logger.error(error_msg)
        return {"error": error_msg, "tool_name": tool_name, "type": "TimeoutError"}
    
    except Exception as e:
        error_msg = f"Erro ao executar ferramenta {tool_name}: {format_exception_message(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback completo: {traceback.format_exc()}")

        return {
            "error": str(e),
            "tool_name": tool_name,
            "type": type(e).__name__,
            "formatted_error": format_exception_message(e)
        }

# --- Teste de Conexão Assíncrono com Retry Logic ---
async def test_mcp_connection_async(server_url: str, max_retries: int = 2) -> tuple[bool, str]:
    """Testa a conexão com o servidor MCP de forma assíncrona com retry logic."""
    for attempt in range(max_retries + 1):
        try:
            # CORREÇÃO: Timeouts mais longos e progressivos
            timeout = 45.0 if attempt == 0 else 90.0  # Aumentado significativamente
            wake_up = attempt > 0
            
            if attempt > 0:
                logger.info(f"Tentativa {attempt + 1} de conexão MCP (timeout: {timeout}s)")
                # CORREÇÃO: Aguardar mais tempo entre tentativas
                await asyncio.sleep(5)
            
            async with mcp_session(server_url, timeout=timeout, wake_up=wake_up) as session:
                # CORREÇÃO: Timeout mais longo para list_tools
                tools = await asyncio.wait_for(session.list_tools(), timeout=30.0)
                success_msg = f"Conexão bem-sucedida. {len(tools.tools)} ferramentas disponíveis."
                if attempt > 0:
                    success_msg += f" (Sucesso na tentativa {attempt + 1})"
                return True, success_msg
                
        except Exception as e:
            error_formatted = format_exception_message(e)
            logger.error(f"Tentativa {attempt + 1} falhou: {error_formatted}")
            
            # Se não é a última tentativa, continua
            if attempt < max_retries:
                wait_time = (attempt + 1) * 5  # CORREÇÃO: Aguardar mais tempo - 5s, 10s
                logger.info(f"Aguardando {wait_time}s antes da próxima tentativa...")
                await asyncio.sleep(wait_time)
                continue
            
            # Última tentativa falhou - formatar erro para o usuário
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['timeout', 'inicialização', 'handshake']):
                return False, f"❌ Timeout na conexão/inicialização. O servidor MCP está respondendo mas não consegue completar a inicialização. Isso é comum em serviços gratuitos. Aguarde 1-2 minutos e tente novamente."
            elif any(keyword in error_str for keyword in ['cold start']):
                return False, f"❌ Servidor em cold start. Aguarde 2-3 minutos e tente novamente."
            elif any(keyword in error_str for keyword in ['connection', 'refused', 'unreachable', '404']):
                return False, f"❌ Servidor MCP inacessível: {server_url}"
            else:
                return False, f"❌ Erro de conexão: {str(e)}"

def test_mcp_connection_sync(server_url: str) -> tuple[bool, str]:
    """Wrapper síncrono para teste de conexão."""
    try:
        return asyncio.run(test_mcp_connection_async(server_url))
    except Exception as e:
        error_formatted = format_exception_message(e)
        logger.error(f"Erro crítico no teste de conexão: {error_formatted}")
        return False, f"❌ Erro crítico: {str(e)}"

# --- Função Principal com Melhor Isolamento de Asyncio ---
async def process_query_with_tools_safe(query: str, gemini_client: genai.Client, sse_url: str):
    """
    Processa uma query usando conexão MCP isolada e o modelo Gemini.
    """
    try:
        # Timeout mais longo para operações de query
        async with mcp_session(sse_url, timeout=120.0, wake_up=False) as session:
            # Obter ferramentas disponíveis
            tools_result = await asyncio.wait_for(session.list_tools(), timeout=30.0)
            available_tool_declarations = []
            for tool in tools_result.tools:
                available_tool_declarations.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                })

            logger.info(f"Ferramentas disponíveis: {[t['name'] for t in available_tool_declarations]}")

            # Prompt de sistema otimizado
            system_instruction = """
            Você é um assistente especializado em dados de energia da U.S. Energy Information Administration (EIA), acessando dados através de uma API v2.
            Seu objetivo é responder às perguntas dos usuários usando as ferramentas fornecidas: `list_eia_v2_routes`, `get_eia_v2_route_data`, e `get_eia_v2_series_id_data`.

            **REGRAS CRÍTICAS PARA USO DAS FERRAMENTAS:**

            1.  **SEMPRE verifique se há erros nos resultados das ferramentas antes de prosseguir.**
            2.  **Se uma ferramenta retornar erro, explique o problema e tente uma abordagem alternativa.**
            3.  **FLUXO DE DESCOBERTA OBRIGATÓRIO (para perguntas abertas sobre dados):**
                *   **NÃO assuma rotas ou IDs.** Para perguntas como "Qual a produção de X em Y?" ou "Dados sobre Z", você DEVE seguir este fluxo:
                *   **Passo 1:** Comece com `list_eia_v2_routes()` (sem argumentos) para ver as categorias de nível superior.
                *   **Passo 2:** Analise a saída. Se uma categoria parecer relevante (ex: "petroleum"), chame `list_eia_v2_routes(segment_path="nome_da_categoria")` para ver sub-rotas e metadados.
                *   **Passo 3:** Continue chamando `list_eia_v2_routes` com `segment_path` cada vez mais específico até encontrar os dados necessários.

            4.  **TRATAMENTO DE ERROS:**
                *   Se encontrar um erro de "TaskGroup" ou similar, informe que houve um problema técnico e tente uma abordagem diferente.
                *   Se uma ferramenta falhar, não desista - tente uma rota alternativa ou explique as limitações.

            5.  **ANO CORRENTE:**
                *   Para "este ano" ou "ano corrente", use 2024 como referência.

            **IMPORTANTE:** Sempre verifique se o resultado de uma ferramenta contém um erro antes de interpretá-lo como dados válidos.
            """

            # Inicializa a sessão de chat
            chat = gemini_client.chats.create(
                model="gemini-1.5-flash-001",
                tools=available_tool_declarations,
                config=types.GenerateContentConfig(temperature=0.1),
                history=[
                    types.Content(
                        role="user",
                        parts=[types.Part(text=system_instruction)]
                    ),
                    types.Content(
                        role="model",
                        parts=[types.Part(text="Olá! Como posso ajudar com os dados de energia da EIA hoje?")]
                    )
                ]
            )
            
            # Processar iterações
            max_iterations = 6  # Reduzido para evitar loops longos
            iteration_count = 0
            final_response_content = "Processamento concluído sem resposta final."

            current_user_message_parts = [types.Part(text=query)]

            while iteration_count < max_iterations:
                iteration_count += 1
                logger.info(f"Iteração {iteration_count} do loop de processamento")
                
                try:
                    # Envia mensagem com timeout
                    gemini_response = await asyncio.wait_for(
                        chat.send_message_async(current_user_message_parts),
                        timeout=60.0
                    )
                    
                    response_text_parts = []
                    tool_calls_for_processing = []

                    for part in gemini_response.parts:
                        if hasattr(part, 'text'):
                            response_text_parts.append(part.text)
                        elif hasattr(part, 'function_call'):
                            tool_calls_for_processing.append(part.function_call)
                    
                    current_llm_text_response = "".join(response_text_parts)

                    if current_llm_text_response:
                        st.markdown(current_llm_text_response)
                        final_response_content = current_llm_text_response

                    if tool_calls_for_processing:
                        tool_outputs_parts = []
                        
                        # Processa as chamadas de ferramenta sequencialmente
                        for tool_call in tool_calls_for_processing:
                            tool_name = tool_call.name
                            try:
                                tool_args = dict(tool_call.args)
                            except Exception as e:
                                st.error(f"Erro ao converter argumentos da ferramenta {tool_name}: {tool_call.args}. Erro: {e}")
                                tool_output_data = {
                                    "error": f"Erro ao converter argumentos: {e}", 
                                    "arguments_received": str(tool_call.args)
                                }
                            else:
                                tool_output_data = await execute_tool_call_safe(session, tool_name, tool_args)
                            
                            # Mostra resultado da ferramenta
                            st.info(f"🔧 Executando: {tool_name}")
                            if isinstance(tool_output_data, dict) and "error" in tool_output_data:
                                st.error(f"❌ Erro na ferramenta: {tool_output_data.get('error', 'Erro desconhecido')}")
                            else:
                                st.success(f"✅ Resultado obtido")
                                with st.expander(f"Ver dados de {tool_name}", expanded=False):
                                    st.json(tool_output_data)

                            tool_outputs_parts.append(
                                types.Part(function_response={
                                    'name': tool_name,
                                    'response': tool_output_data
                                })
                            )
                        
                        current_user_message_parts = tool_outputs_parts
                        
                    else:
                        # Sem mais chamadas de ferramenta - terminar
                        break
                        
                except asyncio.TimeoutError:
                    error_msg = "⏰ Timeout ao comunicar com o modelo Gemini. Tente uma pergunta mais simples."
                    st.error(error_msg)
                    final_response_content = error_msg
                    break
                    
                except types.BlockedPromptException as e:
                    error_msg = f"🚫 A resposta foi bloqueada devido a políticas de segurança: {e.safety_ratings}"
                    st.error(error_msg)
                    logger.error(f"Gemini BlockedPromptException: {e}")
                    final_response_content = error_msg
                    break

                except Exception as e:
                    error_formatted = format_exception_message(e)
                    logger.error(f"Erro na iteração {iteration_count}: {error_formatted}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Verifica se é um erro conhecido de concorrência
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ['taskgroup', 'exceptiongroup', 'unhandled errors']):
                        error_msg = "⚠️ Erro técnico de concorrência detectado. Tente reformular sua pergunta ou tente novamente."
                        st.error(error_msg)
                        with st.expander("Detalhes técnicos", expanded=False):
                            st.text(error_formatted)
                        final_response_content = error_msg
                        break
                    else:
                        # Re-raise outros erros
                        raise

            if iteration_count >= max_iterations:
                warning_msg = "⚠️ Limite máximo de iterações atingido. A resposta pode estar incompleta."
                st.warning(warning_msg)
                final_response_content += f"\n\n{warning_msg}"
            
            return final_response_content
                
    except Exception as e:
        error_formatted = format_exception_message(e)
        logger.error(f"Erro crítico em process_query_with_tools_safe: {error_formatted}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        
        # Categoriza o erro para o usuário
        error_str = str(e).lower()
        if any(keyword in error_str for keyword in ['taskgroup', 'exceptiongroup', 'unhandled errors']):
            user_msg = "❌ Erro de concorrência detectado. Isso pode acontecer em ambientes com múltiplas operações assíncronas. Tente novamente."
        elif any(keyword in error_str for keyword in ['timeout']):
            user_msg = f"❌ Timeout na operação. O servidor pode estar lento ou em cold start."
        elif any(keyword in error_str for keyword in ['connection', 'refused', 'unreachable']):
            user_msg = f"❌ Erro de conexão com o servidor MCP ({MCP_SERVER_URL}). Verifique se o servidor está rodando."
        else:
            user_msg = f"❌ Erro inesperado: {type(e).__name__}: {str(e)}"
        
        raise Exception(user_msg) from e

# --- Cliente Principal ---
class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str, gemini_client: genai.Client):
        self.mcp_server_url = mcp_server_url
        self.gemini_client = gemini_client

    async def process_query(self, query: str):
        """Processa uma query usando conexão MCP isolada."""
        return await process_query_with_tools_safe(query, self.gemini_client, self.mcp_server_url)

# --- Inicialização do Cliente Gemini (primeira coisa a ser feita) ---
if "gemini_client_instance" not in st.session_state:
    try:
        st.session_state.gemini_client_instance = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("Cliente Gemini inicializado com sucesso")
    except Exception as e:
        st.error(f"❌ Erro ao inicializar cliente Gemini: {e}")
        st.stop()

if "mcp_chatbot_client" not in st.session_state:
    st.session_state.mcp_chatbot_client = MCP_ChatBotClient(
        MCP_SERVER_URL,
        st.session_state.gemini_client_instance
    )

# --- Layout Principal e Sidebar ---
col1, col2 = st.columns([3, 1])

# --- Sidebar (sempre visível) ---
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.markdown(f"**🌐 Servidor MCP:** `{MCP_SERVER_URL}`")
    
    # Status da conexão
    if st.session_state.connection_status["status"] == "unknown":
        st.markdown("**📊 Status:** 🟡 Não testado")
    elif st.session_state.connection_status["status"] == "ok":
        st.markdown("**📊 Status:** 🟢 Conectado")
    else:
        st.markdown("**📊 Status:** 🔴 Desconectado")
    
    st.markdown("---")
    
    # Botão para testar conexão
    if st.button("🔍 Testar Conexão MCP"):
        with st.spinner("Testando conexão... (pode demorar se o servidor estiver em cold start)"):
            connection_ok, connection_msg = test_mcp_connection_sync(MCP_SERVER_URL)
            st.session_state.connection_status = {
                "status": "ok" if connection_ok else "error",
                "message": connection_msg
            }
        st.rerun()
    
    # Mostra resultado do último teste
    if st.session_state.connection_status["status"] != "unknown":
        if st.session_state.connection_status["status"] == "ok":
            st.success(st.session_state.connection_status["message"])
        else:
            st.error("Erro na conexão:")
            st.text(st.session_state.connection_status["message"])
    
    # Botão para resetar estado do servidor
    if st.button("🔄 Reset Server State"):
        st.session_state.server_wake_attempted = False
        st.session_state.connection_status = {"status": "unknown", "message": "Estado resetado"}
        st.success("Estado do servidor resetado. Teste a conexão novamente.")
        st.rerun()
    
    if st.button("🗑️ Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔧 Diagnóstico Completo"):
        st.markdown("### 🔍 Executando diagnóstico...")
        
        # Verificar configurações
        if not MCP_SERVER_URL:
            st.error("❌ URL do servidor MCP não configurada")
        else:
            st.success(f"✅ URL configurada")
        
        if not GEMINI_API_KEY:
            st.error("❌ Chave API Gemini não configurada")
        else:
            st.success("✅ Chave API Gemini configurada")
        
        # Testar imports
        try:
            from mcp.client.sse import sse_client
            st.success("✅ Imports MCP OK")
        except ImportError as e:
            st.error(f"❌ Erro de import MCP: {e}")
        
        try:
            import google.genai as genai_test
            test_client = genai_test.Client(api_key="DUMMY_KEY")
            st.success("✅ Imports Google Generative AI OK")
        except Exception as e:
            st.error(f"❌ Erro Google Generative AI: {e}")

# --- Conteúdo Principal ---
with col1:
    # Mostra mensagens do histórico
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Interface de Chat ---
    # Só permite chat se a conexão foi testada e está OK
    if st.session_state.connection_status["status"] == "ok":
        if prompt := st.chat_input("Digite sua pergunta sobre dados de energia..."):
            # Adicionar mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            # Processar resposta do assistente
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analisando sua pergunta e consultando dados..."):
                    try:
                        # Cria uma nova task para isolar o processamento
                        response = asyncio.run(
                            st.session_state.mcp_chatbot_client.process_query(prompt)
                        )
                        
                        if response:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response
                            })
                            
                    except Exception as e:
                        error_formatted = format_exception_message(e)
                        logger.error(f"Erro ao processar consulta: {error_formatted}")
                        
                        # Erro mais user-friendly
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['taskgroup', 'exceptiongroup', 'concorrência']):
                            error_msg = """
                            ⚠️ **Erro de concorrência detectado**
                            
                            Isso pode acontecer devido a operações assíncronas conflitantes. 
                            
                            **Sugestões:**
                            - Tente reformular sua pergunta
                            - Faça perguntas mais específicas
                            - Aguarde alguns segundos antes de tentar novamente
                            """
                        else:
                            error_msg = f"❌ **Erro ao processar consulta:**\n\n{str(e)}"
                        
                        st.error(error_msg)
                        
                        with st.expander("🔍 Detalhes técnicos", expanded=False):
                            st.text(error_formatted)
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    elif st.session_state.connection_status["status"] == "error":
        st.warning("⚠️ **Chatbot indisponível** - Problemas de conexão com o servidor MCP")
        st.info("💡 Use o botão 'Testar Conexão MCP' na barra lateral para verificar o status.")
    
    else:
        st.info("🔍 **Teste a conexão primeiro**")
        st.info("💡 Use o botão 'Testar Conexão MCP' na barra lateral para começar.")

# --- Footer ---
st.markdown("---")
st.markdown("🔗 **EIA Energy Data Chatbot** | Dados em tempo real da U.S. Energy Information Administration")