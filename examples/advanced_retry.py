"""
Exemplo avançado com análise de retry e otimização
"""

import os
import logging
from vizeval import OpenAI, VizevalConfig, Evaluator
from vizeval.exceptions import VizevalThresholdError, VizevalOpenAIError

# Configurar logging detalhado
logging.basicConfig(level=logging.DEBUG)

def main():
    # Configurações para diferentes cenários
    configs = {
        "conservative": VizevalConfig(
            api_key=os.getenv("VIZEVAL_API_KEY"),
            evaluator=Evaluator.MEDICAL,
            threshold=0.9,
            max_retries=5,
            metadata={"strategy": "conservative"}
        ),
        "balanced": VizevalConfig(
            api_key=os.getenv("VIZEVAL_API_KEY"),
            evaluator=Evaluator.MEDICAL,
            threshold=0.8,
            max_retries=3,
            metadata={"strategy": "balanced"}
        ),
        "aggressive": VizevalConfig(
            api_key=os.getenv("VIZEVAL_API_KEY"),
            evaluator=Evaluator.MEDICAL,
            threshold=0.7,
            max_retries=1,
            metadata={"strategy": "aggressive"}
        )
    }
    
    # Perguntas de teste com diferentes níveis de complexidade
    test_questions = [
        {
            "system": "Você é um médico especialista em cardiologia.",
            "user": "Quais são os sintomas de infarto?",
            "complexity": "high"
        },
        {
            "system": "Você é um assistente de saúde.",
            "user": "Como prevenir gripes?",
            "complexity": "low"
        },
        {
            "system": "Você é um farmacêutico.",
            "user": "Quais são as interações do paracetamol?",
            "complexity": "medium"
        }
    ]
    
    print("=== Análise Comparativa de Estratégias ===")
    
    for strategy_name, config in configs.items():
        print(f"\n--- Estratégia: {strategy_name.upper()} ---")
        print(f"Threshold: {config.threshold}, Max Retries: {config.max_retries}")
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            vizeval_config=config
        )
        
        strategy_results = []
        
        for question in test_questions:
            print(f"\nTestando pergunta ({question['complexity']}): {question['user']}")
            
            try:
                result = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": question["system"]},
                        {"role": "user", "content": question["user"]}
                    ],
                    temperature=0.7
                )
                
                # Análise detalhada
                analysis = analyze_retry_result(result)
                strategy_results.append(analysis)
                
                print(f"  ✅ Sucesso: {result.total_attempts} tentativas")
                print(f"  Score final: {result.final_evaluation.score:.3f}")
                print(f"  Passou threshold: {result.passed_threshold}")
                print(f"  Eficiência: {analysis['efficiency']:.2f}%")
                
            except VizevalThresholdError as e:
                print(f"  ❌ Threshold não atingido: {e.best_score}")
                print(f"  Tentativas: {e.attempts}")
                
            except VizevalOpenAIError as e:
                print(f"  ❌ Erro OpenAI: {e}")
        
        # Relatório da estratégia
        print_strategy_report(strategy_name, strategy_results)
    
    # Exemplo de retry inteligente com temperatura adaptativa
    print("\n=== Retry Inteligente com Temperatura Adaptativa ===")
    
    adaptive_config = VizevalConfig(
        api_key=os.getenv("VIZEVAL_API_KEY"),
        evaluator=Evaluator.MEDICAL,
        threshold=0.85,
        max_retries=4,
        metadata={"strategy": "adaptive"}
    )
    
    client_adaptive = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        vizeval_config=adaptive_config
    )
    
    # Pergunta complexa que pode precisar de ajustes
    complex_question = {
        "system": "Você é um médico especialista. Seja preciso e cauteloso.",
        "user": "Quais são os protocolos de tratamento para COVID-19 severa?"
    }
    
    result_adaptive = client_adaptive.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": complex_question["system"]},
            {"role": "user", "content": complex_question["user"]}
        ],
        temperature=0.3  # Começar com temperatura baixa
    )
    
    print("Análise do retry adaptativo:")
    print(f"Tentativas: {result_adaptive.total_attempts}")
    
    # Mostrar evolução dos scores
    for i, attempt in enumerate(result_adaptive.attempts):
        print(f"  Tentativa {i+1}: Score {attempt.score:.3f}")
    
    # Exemplo de otimização de parâmetros
    print("\n=== Otimização de Parâmetros ===")
    
    optimization_results = optimize_parameters(
        client=client_adaptive,
        question=complex_question,
        target_score=0.9
    )
    
    print("Resultados da otimização:")
    for params, score in optimization_results.items():
        print(f"  {params}: {score:.3f}")


def analyze_retry_result(result):
    """Analisa o resultado de um retry para métricas"""
    total_attempts = result.total_attempts
    final_score = result.final_evaluation.score or 0
    passed = result.passed_threshold
    
    # Calcular eficiência (score final / tentativas)
    efficiency = final_score / total_attempts if total_attempts > 0 else 0
    
    # Calcular melhoria entre tentativas
    scores = [attempt.score for attempt in result.attempts if attempt.score is not None]
    improvement = (scores[-1] - scores[0]) / scores[0] if len(scores) > 1 and scores[0] > 0 else 0
    
    return {
        "attempts": total_attempts,
        "final_score": final_score,
        "passed": passed,
        "efficiency": efficiency * 100,
        "improvement": improvement * 100,
        "scores": scores
    }


def print_strategy_report(strategy_name, results):
    """Imprime relatório de uma estratégia"""
    if not results:
        return
    
    total_attempts = sum(r["attempts"] for r in results)
    avg_score = sum(r["final_score"] for r in results) / len(results)
    success_rate = sum(1 for r in results if r["passed"]) / len(results)
    avg_efficiency = sum(r["efficiency"] for r in results) / len(results)
    
    print(f"\n📊 Relatório da Estratégia {strategy_name.upper()}:")
    print(f"  Total de tentativas: {total_attempts}")
    print(f"  Score médio: {avg_score:.3f}")
    print(f"  Taxa de sucesso: {success_rate:.1%}")
    print(f"  Eficiência média: {avg_efficiency:.1f}%")


def optimize_parameters(client, question, target_score):
    """Otimiza parâmetros para atingir um score alvo"""
    parameter_combinations = [
        {"temperature": 0.3, "top_p": 0.9},
        {"temperature": 0.5, "top_p": 0.95},
        {"temperature": 0.7, "top_p": 1.0},
        {"temperature": 0.9, "top_p": 0.8},
    ]
    
    results = {}
    
    for params in parameter_combinations:
        try:
            result = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": question["system"]},
                    {"role": "user", "content": question["user"]}
                ],
                **params
            )
            
            score = result.final_evaluation.score or 0
            params_str = f"temp={params['temperature']}, top_p={params['top_p']}"
            results[params_str] = score
            
        except Exception as e:
            print(f"Erro com parâmetros {params}: {e}")
    
    return results


if __name__ == "__main__":
    main() 