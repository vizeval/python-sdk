"""
Modelos de dados para a SDK Vizeval
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


@dataclass
class VizevalConfig:
    """Configuração para integração com Vizeval"""
    api_key: str
    evaluator: str = "medical"
    threshold: float = 0.8
    max_retries: int = 3
    base_url: str = "https://api.vizeval.com"
    async_mode: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold deve estar entre 0 e 1")
        if self.max_retries < 0:
            raise ValueError("max_retries deve ser maior ou igual a 0")


class EvaluationRequest(BaseModel):
    """Requisição de avaliação para a API Vizeval"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    system_prompt: str
    user_prompt: str
    response: str
    evaluator: str
    metadata: Dict[str, Any] = {}
    api_key: str
    async_mode: bool = False


class EvaluationResponse(BaseModel):
    """Resposta de avaliação da API Vizeval"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    evaluator: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Verifica se a avaliação foi bem-sucedida"""
        return self.score is not None
    
    def passed_threshold(self, threshold: float = 0.8) -> bool:
        """Verifica se passou do threshold (configurável)"""
        return self.score is not None and self.score >= threshold


@dataclass
class RetryAttempt:
    """Informações sobre uma tentativa de retry"""
    attempt_number: int
    score: Optional[float]
    feedback: Optional[str]
    openai_response: Any
    evaluation_response: EvaluationResponse
    
    
@dataclass
class VizevalResult:
    """Resultado completo da avaliação Vizeval com histórico de tentativas"""
    final_response: Any  # Resposta final do OpenAI
    final_evaluation: EvaluationResponse
    attempts: list[RetryAttempt]
    config: VizevalConfig
    
    @property
    def total_attempts(self) -> int:
        return len(self.attempts)
    
    @property
    def passed_threshold(self) -> bool:
        return self.final_evaluation.passed_threshold
    
    @property
    def best_score(self) -> Optional[float]:
        """Melhor score obtido em todas as tentativas"""
        scores = [attempt.score for attempt in self.attempts if attempt.score is not None]
        return max(scores) if scores else None 