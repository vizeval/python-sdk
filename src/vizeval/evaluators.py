"""
Constantes e enums para evaluators Vizeval
"""

from enum import Enum
from typing import List


class Evaluator(str, Enum):
    """Evaluators disponíveis na API Vizeval"""
    MEDICAL = "medical"
    JURIDICAL = "juridical"
    DUMMY = "dummy"


# Lista de evaluators disponíveis
AVAILABLE_EVALUATORS: List[str] = [e.value for e in Evaluator]

# Configurações default por evaluator
EVALUATOR_DEFAULTS = {
    Evaluator.MEDICAL: {
        "threshold": 0.8,
        "description": "Avaliação especializada em conteúdo médico e de saúde",
        "focus": ["alucinações médicas", "informações factualmente incorretas", "conselhos médicos perigosos"]
    },
    Evaluator.JURIDICAL: {
        "threshold": 0.7,
        "description": "Avaliação especializada em conteúdo jurídico",
        "focus": ["precisão legal", "jurisprudência", "interpretação de leis"]
    },
    Evaluator.DUMMY: {
        "threshold": 0.5,
        "description": "Evaluator de teste com avaliação aleatória",
        "focus": ["teste", "desenvolvimento"]
    }
}


def get_evaluator_info(evaluator: str) -> dict:
    """Obtém informações sobre um evaluator específico"""
    if evaluator not in AVAILABLE_EVALUATORS:
        raise ValueError(f"Evaluator '{evaluator}' não disponível. Disponíveis: {AVAILABLE_EVALUATORS}")
    
    return EVALUATOR_DEFAULTS.get(evaluator, {})


def validate_evaluator(evaluator: str) -> bool:
    """Valida se um evaluator é válido"""
    return evaluator in AVAILABLE_EVALUATORS 