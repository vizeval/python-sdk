"""
Exemplo básico de uso da SDK Vizeval
"""

import os
from vizeval import VizevalClient, Evaluator

def main():
    # Configuração básica
    client = VizevalClient(
        api_key=os.getenv("VIZEVAL_API_KEY"),
        base_url="https://api.vizeval.com"
    )
    
    # Exemplo 1: Avaliação simples
    print("=== Avaliação Simples ===")
    
    system_prompt = "Você é um assistente médico especializado em fornecer informações precisas sobre saúde."
    user_prompt = "Quais são os sintomas da gripe?"
    response = """Os sintomas da gripe incluem:
    - Febre alta (acima de 38°C)
    - Dor de cabeça
    - Dores musculares
    - Fadiga
    - Tosse seca
    - Congestão nasal
    
    É importante procurar um médico se os sintomas persistirem por mais de 7 dias."""
    
    evaluation = client.evaluate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        evaluator=Evaluator.MEDICAL
    )
    
    print(f"Score: {evaluation.score}")
    print(f"Feedback: {evaluation.feedback}")
    print(f"Evaluator: {evaluation.evaluator}")
    
    # Exemplo 2: Avaliação com diferentes evaluators
    print("\n=== Comparação de Evaluators ===")
    
    evaluators = [Evaluator.MEDICAL, Evaluator.DUMMY]
    
    for evaluator in evaluators:
        evaluation = client.evaluate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            evaluator=evaluator
        )
        
        print(f"\n{evaluator.value}:")
        print(f"  Score: {evaluation.score}")
        print(f"  Passou threshold: {evaluation.passed_threshold}")
    
    # Exemplo 3: Avaliação com metadados
    print("\n=== Avaliação com Metadados ===")
    
    metadata = {
        "model": "gpt-4",
        "temperature": 0.7,
        "domain": "medicina_geral",
        "user_id": "user123"
    }
    
    evaluation = client.evaluate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        evaluator=Evaluator.MEDICAL,
        metadata=metadata
    )
    
    print(f"Score: {evaluation.score}")
    print(f"Avaliação com contexto adicional registrado")
    
    # Exemplo 4: Health check
    print("\n=== Health Check ===")
    
    if client.health_check():
        print("✅ API Vizeval está funcionando")
    else:
        print("❌ API Vizeval não está acessível")
    
    # Exemplo 5: Listar avaliações do usuário
    print("\n=== Histórico de Avaliações ===")
    
    try:
        evaluations = client.get_user_evaluations()
        print(f"Total de avaliações: {len(evaluations)}")
        
        if evaluations:
            latest = evaluations[-1]
            print(f"Última avaliação: {latest.get('evaluator')} - Score: {latest.get('score')}")
    except Exception as e:
        print(f"Erro ao obter avaliações: {e}")
    
    # Fechar conexão
    client.close()


if __name__ == "__main__":
    main() 