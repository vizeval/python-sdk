"""
Exemplo de integração transparente com OpenAI
"""

import os
import logging
from vizeval import OpenAI, VizevalConfig, Evaluator

# Configurar logging para ver o processo
logging.basicConfig(level=logging.INFO)

def main():
    # Configuração Vizeval
    vizeval_config = VizevalConfig(
        api_key=os.getenv("VIZEVAL_API_KEY"),
        evaluator=Evaluator.MEDICAL,
        threshold=0.8,
        max_retries=3,
        base_url="https://api.vizeval.com"
    )
    
    # Exemplo 1: Integração básica transparente
    print("=== Integração Transparente com OpenAI ===")
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        vizeval_config=vizeval_config
    )
    
    # Chamada normal que será automaticamente avaliada e retriada se necessário
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente médico especializado."},
            {"role": "user", "content": "Qual é o tratamento para hipertensão?"}
        ],
        temperature=0.7
    )
    
    # O resultado é um VizevalResult com informações completas
    print(f"Tentativas totais: {result.total_attempts}")
    print(f"Passou do threshold: {result.passed_threshold}")
    print(f"Score final: {result.final_evaluation.score}")
    print(f"Feedback: {result.final_evaluation.feedback}")
    
    # Acessar a resposta original do OpenAI
    openai_response = result.final_response
    print(f"\nResposta: {openai_response.choices[0].message.content}")
    
    # Exemplo 2: Uso com configuração dinâmica
    print("\n=== Configuração Dinâmica ===")
    
    # Criar cliente sem configuração inicial
    client2 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Definir configuração dinamicamente
    client2.set_vizeval_config({
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": "medical",
        "threshold": 0.9,  # Threshold mais alto
        "max_retries": 2,
        "metadata": {"experiment": "high_threshold_test"}
    })
    
    result2 = client2.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Quais são os efeitos colaterais do ibuprofeno?"}
        ]
    )
    
    print(f"Tentativas: {result2.total_attempts}")
    print(f"Melhor score: {result2.best_score}")
    
    # Exemplo 3: Desabilitar Vizeval temporariamente
    print("\n=== Desabilitando Vizeval ===")
    
    client2.disable_vizeval()
    
    # Esta chamada usará apenas OpenAI, sem avaliação
    normal_response = client2.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Olá!"}
        ]
    )
    
    print(f"Resposta normal (sem Vizeval): {type(normal_response)}")
    
    # Exemplo 4: Histórico de tentativas
    print("\n=== Análise de Tentativas ===")
    
    client.set_vizeval_config({
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": "medical",
        "threshold": 0.95,  # Threshold muito alto para forçar retries
        "max_retries": 3
    })
    
    try:
        result3 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Como tratar uma dor de cabeça?"}
            ],
            temperature=0.3
        )
        
        print(f"Análise das {result3.total_attempts} tentativas:")
        for i, attempt in enumerate(result3.attempts):
            print(f"  Tentativa {attempt.attempt_number}: Score {attempt.score}")
            
    except Exception as e:
        print(f"Erro após múltiplas tentativas: {e}")
    
    # Exemplo 5: Configuração por dict
    print("\n=== Configuração por Dict ===")
    
    client3 = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        vizeval_config={
            "api_key": os.getenv("VIZEVAL_API_KEY"),
            "evaluator": "medical",
            "threshold": 0.7,
            "max_retries": 1,
            "metadata": {"source": "dict_config"}
        }
    )
    
    result4 = client3.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "O que é diabetes?"}
        ]
    )
    
    print(f"Configuração por dict funcionou: {result4.passed_threshold}")


if __name__ == "__main__":
    main() 