import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

# Carregar a chave da API Groq das variáveis de ambiente (configuradas no Vercel)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    # Isso não vai parar o servidor Vercel, mas as chamadas falharão.
    # Idealmente, Vercel não deveria nem buildar sem a variável.
    print("ALERTA: GROQ_API_KEY não está configurada como variável de ambiente!")

client = Groq(api_key=GROQ_API_KEY)

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    model: str = "llama3-70b-8192" # Modelo padrão, pode ser sobrescrito na requisição

class GroqResponse(BaseModel):
    answer: str

@app.post("/ask-groq", response_model=GroqResponse)
async def ask_groq_endpoint(request_data: QuestionRequest):
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY não configurada no servidor.")

    system_prompt = (
        "Você é um assistente especializado em Inteligência Artificial. "
        "Sua tarefa é explicar conceitos de IA de forma extremamente simples, clara e concisa "
        "para um público leigo, que não tem conhecimento técnico nem de programação. "
        "Evite jargões técnicos o máximo possível. Se precisar usar um termo técnico, explique-o brevemente. "
        "Use analogias do dia a dia, se apropriado. O objetivo é desmistificar a IA."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": request_data.question,
                }
            ],
            model=request_data.model,
        )
        answer = chat_completion.choices[0].message.content
        return GroqResponse(answer=answer)
    except Exception as e:
        print(f"Erro ao chamar a API Groq: {e}") # Log no Vercel
        raise HTTPException(status_code=500, detail=f"Erro ao processar sua pergunta com Groq: {str(e)}")

@app.get("/")
async def read_root():
    return {"message": "API Groq para PowerPoint está no ar!"}

# Para testar localmente (opcional, Vercel usa seu próprio servidor)
# if __name__ == "__main__":
#     import uvicorn
#     if not GROQ_API_KEY:
#        print("Para rodar localmente, defina a variável de ambiente GROQ_API_KEY.")
#        # Exemplo: export GROQ_API_KEY="sua_chave_aqui" no terminal antes de rodar
#     uvicorn.run(app, host="0.0.0.0", port=8000)