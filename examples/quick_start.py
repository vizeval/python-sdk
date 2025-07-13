"""
Exemplo de início rápido da SDK Vizeval
Demonstra como substituir OpenAI por Vizeval com mudanças mínimas
"""

import os
from vizeval import OpenAI

def main():
    print("🚀 Vizeval SDK - Início Rápido")
    print("=" * 50)
    
    # ANTES: Código OpenAI padrão
    print("\n📋 Código OpenAI Original:")
    print("""
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente médico."},
            {"role": "user", "content": "Quais são os sintomas da gripe?"}
        ]
    )
    
    print(response.choices[0].message.content)
    """)
    
    # DEPOIS: Código com Vizeval (mudanças mínimas)
    print("\n✨ Código com Vizeval (apenas adicionar configuração):")
    print("""
    from vizeval import OpenAI  # Única mudança no import
    
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        vizeval_config={                      # Adicionar configuração
            "api_key": os.getenv("VIZEVAL_API_KEY"),
            "evaluator": "medical",
            "threshold": 0.8,
            "max_retries": 3
        }
    )
    
    result = client.chat.completions.create(  # Mesma chamada
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um assistente médico."},
            {"role": "user", "content": "Quais são os sintomas da gripe?"}
        ]
    )
    
    # Agora você tem informações de avaliação!
    print(f"Score: {result.final_evaluation.score}")
    print(f"Tentativas: {result.total_attempts}")
    print(result.final_response.choices[0].message.content)
    """)
    
    # Demonstração prática
    print("\n🎯 Demonstração Prática:")
    print("-" * 30)
    
    # Verificar variáveis de ambiente
    if not os.getenv("VIZEVAL_API_KEY"):
        print("⚠️  Configure VIZEVAL_API_KEY para executar a demonstração")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Configure OPENAI_API_KEY para executar a demonstração")
        return
    
    # Criar cliente Vizeval
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        vizeval_config={
            "api_key": os.getenv("VIZEVAL_API_KEY"),
            "evaluator": "medical",
            "threshold": 0.8,
            "max_retries": 3,
            "metadata": {"demo": "quick_start"}
        }
    )
    
    # Exemplos de perguntas médicas
    questions = [
        "Quais são os sintomas da gripe?",
        "Como tratar uma dor de cabeça?",
        "O que é hipertensão arterial?",
        "Quais são os efeitos colaterais do paracetamol?"
    ]
    
    print(f"\n🔍 Testando {len(questions)} perguntas médicas...")
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Pergunta {i}: {question}")
        
        try:
            result = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Você é um assistente médico especializado."},
                    {"role": "user", "content": question}
                ],
                temperature=0.7
            )
            
            # Mostrar informações de avaliação
            print(f"   ✅ Score: {result.final_evaluation.score:.3f}")
            print(f"   🔄 Tentativas: {result.total_attempts}")
            print(f"   🎯 Passou threshold: {'Sim' if result.passed_threshold else 'Não'}")
            
            # Mostrar resposta (primeiras 100 chars)
            response_text = result.final_response.choices[0].message.content
            preview = response_text[:100] + "..." if len(response_text) > 100 else response_text
            print(f"   💬 Resposta: {preview}")
            
        except Exception as e:
            print(f"   ❌ Erro: {str(e)}")
    
    # Demonstração de configuração dinâmica
    print("\n🔧 Demonstração de Configuração Dinâmica:")
    print("-" * 45)
    
    # Testar com threshold alto
    print("\n📈 Testando com threshold alto (0.9)...")
    client.set_vizeval_config({
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": "medical",
        "threshold": 0.9,  # Threshold alto
        "max_retries": 5
    })
    
    result = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Você é um médico especialista."},
            {"role": "user", "content": "Explique detalhadamente sobre diabetes tipo 2."}
        ],
        temperature=0.5
    )
    
    print(f"   Score final: {result.final_evaluation.score:.3f}")
    print(f"   Tentativas necessárias: {result.total_attempts}")
    print(f"   Histórico de scores: {[f'{a.score:.3f}' for a in result.attempts]}")
    
    # Desabilitar Vizeval temporariamente
    print("\n🔇 Desabilitando Vizeval temporariamente...")
    client.disable_vizeval()
    
    normal_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "Olá, como você está?"}
        ]
    )
    
    print(f"   Resposta sem Vizeval: {type(normal_response).__name__}")
    
    print("\n✨ Demonstração concluída!")
    print("🎉 A SDK Vizeval adiciona avaliação especializada com mudanças mínimas!")
    print("\n💡 Próximos passos:")
    print("   1. Veja examples/openai_integration.py para mais exemplos")
    print("   2. Veja examples/advanced_retry.py para configurações avançadas")
    print("   3. Consulte a documentação em https://docs.vizeval.com")


if __name__ == "__main__":
    main() 