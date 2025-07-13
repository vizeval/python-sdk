"""
Vizeval SDK - Avaliação especializada de LLMs na área da saúde

Esta SDK permite integração transparente com a API Vizeval para avaliar
respostas de LLMs com foco em conteúdo médico e de saúde.
"""

from .client import VizevalClient
from .openai_wrapper import OpenAI
from .models import EvaluationRequest, EvaluationResponse, VizevalConfig
from .evaluators import Evaluator
from .exceptions import VizevalError, VizevalAPIError, VizevalConfigError

__version__ = "0.1.1"
__all__ = [
    "VizevalClient",
    "OpenAI",
    "EvaluationRequest",
    "EvaluationResponse", 
    "VizevalConfig",
    "Evaluator",
    "VizevalError",
    "VizevalAPIError",
    "VizevalConfigError",
] 