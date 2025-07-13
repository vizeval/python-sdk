"""
Exceções customizadas para a SDK Vizeval
"""


class VizevalError(Exception):
    """Exceção base para erros da SDK Vizeval"""
    pass


class VizevalAPIError(VizevalError):
    """Erro na comunicação com a API Vizeval"""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}
        

class VizevalConfigError(VizevalError):
    """Erro na configuração da SDK"""
    pass


class VizevalThresholdError(VizevalError):
    """Erro quando não consegue atingir o threshold após max_retries"""
    
    def __init__(self, message: str, best_score: float = None, attempts: int = 0):
        super().__init__(message)
        self.best_score = best_score
        self.attempts = attempts


class VizevalOpenAIError(VizevalError):
    """Erro na integração com OpenAI"""
    pass 