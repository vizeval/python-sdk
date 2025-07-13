# Vizeval SDK

SDK Python para integração transparente com a API Vizeval - Avaliação especializada de LLMs na área da saúde.

## 🎯 Visão Geral

A Vizeval SDK permite integrar facilmente a avaliação especializada de respostas de LLMs em aplicações Python, com foco particular em conteúdo médico e de saúde. A SDK oferece:

- **Integração transparente** com OpenAI
- **Retry automático** baseado em threshold configurável
- **Avaliação especializada** por evaluators médicos
- **Implementação simples** com mudanças mínimas no código existente

## 📦 Instalação

```bash
pip install vizeval
```

## 🚀 Uso Rápido

### Integração Transparente com OpenAI

```python
from vizeval import OpenAI
import os

# Substitua OpenAI padrão por Vizeval OpenAI
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    vizeval_config={
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": "medical",
        "threshold": 0.8,
        "max_retries": 3
    }
)

# Chamada normal - será automaticamente avaliada e retriada se necessário
result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Você é um assistente médico especializado."},
        {"role": "user", "content": "Quais são os sintomas da gripe?"}
    ]
)

# Resultado contém informações de avaliação
print(f"Score: {result.final_evaluation.score}")
print(f"Tentativas: {result.total_attempts}")
print(f"Resposta: {result.final_response.choices[0].message.content}")
```

### Uso Direto da API

```python
from vizeval import VizevalClient

client = VizevalClient(api_key=os.getenv("VIZEVAL_API_KEY"))

evaluation = client.evaluate(
    system_prompt="Você é um assistente médico especializado.",
    user_prompt="Quais são os sintomas da gripe?",
    response="Os sintomas incluem febre, dor de cabeça, fadiga...",
    evaluator="medical"
)

print(f"Score: {evaluation.score}")
print(f"Feedback: {evaluation.feedback}")
```

## 🔧 Configuração

### VizevalConfig

```python
from vizeval import VizevalConfig, Evaluator

config = VizevalConfig(
    api_key="sua_vizeval_api_key",
    evaluator=Evaluator.MEDICAL,  # ou "medical"
    threshold=0.8,                # Score mínimo aceitável
    max_retries=3,                # Máximo de tentativas
    base_url="https://api.vizeval.com",
    metadata={"user_id": "123"}   # Metadados opcionais
)
```

## 🔄 Sistema de Retry

A SDK implementa retry automático inteligente:

1. **Avaliação inicial**: Resposta é avaliada automaticamente
2. **Verificação de threshold**: Se score < threshold, nova tentativa

### Exemplo de Retry Customizado

```python
from vizeval import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    vizeval_config={
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": "medical",
        "threshold": 0.9,     # Threshold alto
        "max_retries": 5,     # Mais tentativas
    }
)

result = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Pergunta médica complexa"}],
    temperature=0.3  # Começar com baixa temperature
)

# Analisar tentativas
for attempt in result.attempts:
    print(f"Tentativa {attempt.attempt_number}: Score {attempt.score}")
```

## 🔍 Funcionalidades Avançadas

### Configuração Dinâmica

```python
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configurar dinamicamente
client.set_vizeval_config({
    "api_key": os.getenv("VIZEVAL_API_KEY"),
    "evaluator": "medical",
    "threshold": 0.8,
    "max_retries": 3
})

# Desabilitar temporariamente
client.disable_vizeval()
```

### Análise de Resultados

```python
result = client.chat.completions.create(...)

# Informações detalhadas
print(f"Total de tentativas: {result.total_attempts}")
print(f"Passou do threshold: {result.passed_threshold}")
print(f"Melhor score: {result.best_score}")
print(f"Score final: {result.final_evaluation.score}")
print(f"Feedback: {result.final_evaluation.feedback}")

# Histórico de tentativas
for i, attempt in enumerate(result.attempts):
    print(f"Tentativa {i+1}: {attempt.score}")
```

### Tratamento de Erros

```python
from vizeval.exceptions import VizevalThresholdError, VizevalAPIError

try:
    result = client.chat.completions.create(...)
except VizevalThresholdError as e:
    print(f"Threshold não atingido após {e.attempts} tentativas")
    print(f"Melhor score: {e.best_score}")
except VizevalAPIError as e:
    print(f"Erro na API: {e}")
```

## 📊 Exemplos Práticos

### Exemplo 1: Avaliação Médica Básica

```python
from vizeval import OpenAI, Evaluator

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    vizeval_config={
        "api_key": os.getenv("VIZEVAL_API_KEY"),
        "evaluator": Evaluator.MEDICAL,
        "threshold": 0.8,
        "max_retries": 3
    }
)

result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Você é um médico especialista."},
        {"role": "user", "content": "Quais são os sintomas de diabetes?"}
    ]
)
```

## 🔧 Desenvolvimento

### Instalação para Desenvolvimento

```bash
git clone https://github.com/vizeval/python-sdk
cd python-sdk
pip install -e ".[dev]"
```

### Executar Testes

```bash
pytest tests/
```

### Formatação de Código

```bash
black src/
isort src/
flake8 src/
```

## 🌟 Recursos Principais

- ✅ **Integração Transparente**: Substitui OpenAI sem mudanças no código
- ✅ **Retry Inteligente**: Ajusta parâmetros automaticamente
- ✅ **Avaliação Especializada**: Evaluators médicos especializados
- ✅ **Configuração Flexível**: Threshold e retries configuráveis
- ✅ **Análise Detalhada**: Histórico completo de tentativas
- ✅ **Tratamento de Erros**: Exceções específicas e informativas
- ✅ **Logging Completo**: Visibilidade total do processo
- ✅ **Fallback Inteligente**: Sempre retorna a melhor resposta

## 🔗 Links Úteis

- [Exemplos no GitHub](https://github.com/vizeval/python-sdk/tree/main/examples)
- [Issues](https://github.com/vizeval/python-sdk/issues)

---

**Desenvolvido com ❤️ para o Hackathon da Adapta**
