import streamlit as st
import asyncio
import json
import os
from typing import List, Dict, TypedDict, Any, Optional, Union
import logging
import traceback

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

# --- Função para Executar Tool Call com Melhor Tratamento de Erros ---
async def execute_tool_call(session: ClientSession, tool_name: str, tool_args: dict):
    """Executa uma chamada de ferramenta no servidor MCP com tratamento robusto de erros."""
    try:
        logger.info(f"Executando ferramenta: {tool_name} com argumentos: {tool_args}")
        result = await session.call_tool(tool_name, arguments=tool_args)
        
        # Formatar resultado de forma mais robusta
        if hasattr(result, 'content') and result.content:
            tool_output_content = []
            for item in result.content:
                if hasattr(item, 'dict'):
                    tool_output_content.append(item.dict())
                elif hasattr(item, '__dict__'):
                    tool_output_content.append(vars(item))
                else:
                    tool_output_content.append(str(item))
            
            return json.dumps(tool_output_content, indent=2, ensure_ascii=False)
        else:
            return json.dumps({"result": "Nenhum conteúdo retornado", "tool_name": tool_name})
        
    except Exception as e:
        logger.error(f"Erro ao executar ferramenta {tool_name}: {e}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        
        # Tratamento especial para TaskGroup/ExceptionGroup
        error_details = {"error": str(e), "tool_name": tool_name, "type": type(e).__name__}
        
        if hasattr(e, 'exceptions') and e.exceptions:
            error_details["sub_exceptions"] = []
            for i, sub_e in enumerate(e.exceptions):
                error_details["sub_exceptions"].append({
                    "index": i,
                    "type": type(sub_e).__name__,
                    "message": str(sub_e)
                })
        
        return json.dumps(error_details, indent=2, ensure_ascii=False)

# --- Função Principal de Processamento com Tratamento Melhorado ---
async def process_query_with_tools(query: str, groq_client: Groq, sse_url: str):
    """
    Processa uma query usando o padrão correto de conexão MCP com tratamento robusto de erros.
    """
    try:
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

                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': query}
                ]
                
                response_placeholder = st.empty()
                max_iterations = 10  # Limite para evitar loops infinitos
                iteration_count = 0
                
                while iteration_count < max_iterations:
                    iteration_count += 1
                    logger.info(f"Iteração {iteration_count} do loop de processamento")
                    
                    try:
                        # Chamada para o Groq
                        chat_completion = groq_client.chat.completions.create(
                            messages=messages,
                            model="llama3-70b-8192", 
                            tools=available_tools,
                            tool_choice="auto",
                            max_tokens=4096,
                            temperature=0.1  # Reduzir aleatoriedade para mais consistência
                        )
                        
                        response_message = chat_completion.choices[0].message
                        logger.info(f"LLM Response (iteração {iteration_count}): {response_message}")

                        if response_message.content:
                            response_placeholder.markdown(response_message.content)
                        
                        tool_calls = response_message.tool_calls
                        
                        if tool_calls:
                            # Adicionar a resposta do assistente às mensagens
                            messages.append({
                                "role": "assistant",
                                "content": response_message.content,
                                "tool_calls": [
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    } for tc in tool_calls
                                ]
                            })
                            
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
                                    continue

                                st.info(f"Executando ferramenta: {tool_name}")
                                if tool_args:  # Só mostrar se não estiver vazio
                                    st.json(tool_args)
                                
                                # Executar a ferramenta com tratamento robusto de erros
                                tool_output = await execute_tool_call(session, tool_name, tool_args)
                                
                                # Mostrar resultado da ferramenta
                                try:
                                    tool_output_json = json.loads(tool_output)
                                    st.success(f"Resultado da ferramenta {tool_name}:")
                                    
                                    # Verificar se há erros no resultado
                                    if isinstance(tool_output_json, dict) and "error" in tool_output_json:
                                        st.error(f"Erro na ferramenta: {tool_output_json.get('error', 'Erro desconhecido')}")
                                    else:
                                        st.json(tool_output_json)
                                        
                                except json.JSONDecodeError:
                                    st.success(f"Resultado da ferramenta {tool_name} (texto):")
                                    st.text(tool_output)

                                messages.append({
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": tool_output,
                                })
                        else:
                            # Se não houver tool_calls, a conversa terminou
                            break
                            
                    except Exception as e:
                        logger.error(f"Erro na iteração {iteration_count}: {e}")
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        
                        # Tratamento específico para TaskGroup/ExceptionGroup
                        if "TaskGroup" in str(e) or hasattr(e, 'exceptions'):
                            error_msg = f"Erro técnico detectado (TaskGroup): {str(e)}"
                            if hasattr(e, 'exceptions') and e.exceptions:
                                error_msg += f"\nSub-erros: {[str(sub_e) for sub_e in e.exceptions]}"
                            
                            st.error(error_msg)
                            return f"Desculpe, houve um problema técnico ao processar sua solicitação. Erro: {str(e)}"
                        else:
                            raise  # Re-raise outros tipos de erro
                
                if iteration_count >= max_iterations:
                    st.warning("Limite máximo de iterações atingido. A resposta pode estar incompleta.")
                
                return response_message.content if response_message.content else "Processamento concluído sem resposta final."
                
    except Exception as e:
        logger.error(f"Erro crítico em process_query_with_tools: {e}")
        logger.error(f"Traceback completo: {traceback.format_exc()}")
        
        # Tratamento específico para diferentes tipos de erro
        if "TaskGroup" in str(e) or hasattr(e, 'exceptions'):
            error_msg = f"Erro de concorrência detectado: {str(e)}"
            if hasattr(e, 'exceptions') and e.exceptions:
                error_msg += f"\nErros subjacentes: {[f'{type(sub_e).__name__}: {str(sub_e)}' for sub_e in e.exceptions]}"
        else:
            error_msg = f"Erro inesperado: {type(e).__name__}: {str(e)}"
        
        raise Exception(error_msg) from e

# --- Cliente Principal Simplificado ---
class MCP_ChatBotClient:
    def __init__(self, mcp_server_url: str, groq_client: Groq):
        self.mcp_server_url = mcp_server_url
        self.groq_client = groq_client

    async def process_query(self, query: str):
        """Processa uma query usando conexão MCP adequada."""
        return await process_query_with_tools(query, self.groq_client, self.mcp_server_url)

# --- Teste de Conexão Melhorado ---
@st.cache_data(ttl=300)
def test_mcp_connection(server_url: str) -> tuple[bool, str]:
    """Testa a conexão com o servidor MCP com tratamento robusto de erros."""
    async def _test():
        try:
            async with sse_client(server_url) as (in_stream, out_stream):
                async with ClientSession(in_stream, out_stream) as session:
                    await session.initialize()
                    return True, "Conexão bem-sucedida"
        except Exception as e:
            logger.error(f"Erro de conexão: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            error_msg = f"Erro de conexão: {type(e).__name__}: {str(e)}"
            
            # Tratamento especial para TaskGroup/ExceptionGroup
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
        logger.error(f"Erro crítico no teste de conexão: {e}")
        return False, f"Erro crítico ao testar conexão: {type(e).__name__}: {str(e)}"

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
                    response = asyncio.run(
                        st.session_state.mcp_chatbot_client.process_query(prompt)
                    )
                    
                    if response:
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                except Exception as e:
                    logger.error(f"Erro ao processar consulta: {e}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    
                    # Tratamento mais específico dos erros
                    if "TaskGroup" in str(e) or hasattr(e, 'exceptions'):
                        error_msg = "❌ Erro técnico detectado (problema de concorrência). Tente reformular sua pergunta ou tente novamente."
                        if hasattr(e, 'exceptions') and e.exceptions:
                            error_msg += f"\n\n**Detalhes técnicos:** {[str(sub_e) for sub_e in e.exceptions]}"
                    else:
                        error_msg = f"❌ Erro ao processar consulta: {str(e)}"
                    
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
        test_mcp_connection.clear()
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
    
    # Informações de depuração
    st.markdown("### Informações de Debug")
    st.markdown(f"**Python:** {os.sys.version}")
    st.markdown(f"**Asyncio Policy:** {type(asyncio.get_event_loop_policy()).__name__}")
    
    if st.button("🐛 Log de Debug"):
        st.write("Últimas mensagens de log:")
        # Aqui você poderia mostrar logs mais detalhados se necessário
        st.info("Logs detalhados estão sendo escritos no console/terminal onde o Streamlit está rodando.")