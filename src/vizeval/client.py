"""
Cliente principal para comunicação com a API Vizeval
"""

import requests
import json
from typing import Optional, Dict, Any
from urllib.parse import urljoin

from .models import EvaluationRequest, EvaluationResponse, VizevalConfig
from .exceptions import VizevalAPIError, VizevalConfigError
from .evaluators import validate_evaluator


class VizevalClient:
    """Cliente para interagir com a API Vizeval"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.vizeval.com"):
        """
        Inicializa o cliente Vizeval
        
        Args:
            api_key: Chave de API do Vizeval
            base_url: URL base da API Vizeval
        """
        if not api_key:
            raise VizevalConfigError("api_key é obrigatório")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Vizeval-SDK/0.1.0'
        })
    
    def evaluate(
        self, 
        system_prompt: str,
        user_prompt: str,
        response: str,
        evaluator: str = "medical",
        metadata: Optional[Dict[str, Any]] = None,
        async_mode: bool = False
    ) -> EvaluationResponse:
        """
        Avalia uma resposta usando a API Vizeval
        
        Args:
            system_prompt: Prompt do sistema usado na geração
            user_prompt: Prompt do usuário
            response: Resposta do LLM a ser avaliada
            evaluator: Tipo de evaluator a ser usado
            metadata: Metadados adicionais
            async_mode: Se deve usar modo assíncrono
            
        Returns:
            EvaluationResponse com o resultado da avaliação
            
        Raises:
            VizevalAPIError: Se ocorrer erro na API
            VizevalConfigError: Se a configuração estiver inválida
        """
        if not validate_evaluator(evaluator):
            raise VizevalConfigError(f"Evaluator '{evaluator}' não é válido")
        
        request_data = EvaluationRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            evaluator=evaluator,
            metadata=metadata or {},
            api_key=self.api_key,
            async_mode=async_mode
        )
        
        return self._make_evaluation_request(request_data)
    
    def _make_evaluation_request(self, request_data: EvaluationRequest) -> EvaluationResponse:
        """
        Faz a requisição para a API de avaliação
        
        Args:
            request_data: Dados da requisição
            
        Returns:
            EvaluationResponse com o resultado
            
        Raises:
            VizevalAPIError: Se ocorrer erro na API
        """
        url = urljoin(self.base_url, "/evaluation/")
        
        try:
            response = self.session.post(
                url,
                data=request_data.model_dump_json(),
                timeout=30
            )
            
            # Trata erros HTTP
            if response.status_code != 201:
                error_message = f"Erro na API Vizeval: {response.status_code}"
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        error_message = f"{error_message} - {error_data['detail']}"
                except:
                    pass
                
                raise VizevalAPIError(
                    error_message,
                    status_code=response.status_code,
                    response_data=response.json() if response.content else {}
                )
            
            # Parse da resposta
            response_data = response.json()
            return EvaluationResponse(**response_data)
            
        except requests.exceptions.RequestException as e:
            raise VizevalAPIError(f"Erro de conexão com a API Vizeval: {str(e)}")
        except json.JSONDecodeError as e:
            raise VizevalAPIError(f"Erro ao parsear resposta da API: {str(e)}")
    
    def get_user_evaluations(self) -> list:
        """
        Obtém todas as avaliações do usuário
        
        Returns:
            Lista de avaliações do usuário
            
        Raises:
            VizevalAPIError: Se ocorrer erro na API
        """
        url = urljoin(self.base_url, "/user/evaluations")
        
        try:
            response = self.session.get(
                url,
                params={"api_key": self.api_key},
                timeout=30
            )
            
            if response.status_code != 200:
                raise VizevalAPIError(
                    f"Erro ao obter avaliações: {response.status_code}",
                    status_code=response.status_code,
                    response_data=response.json() if response.content else {}
                )
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise VizevalAPIError(f"Erro de conexão com a API Vizeval: {str(e)}")
    
    def health_check(self) -> bool:
        """
        Verifica se a API está funcionando
        
        Returns:
            True se a API estiver funcionando
        """
        try:
            url = urljoin(self.base_url, "/health")
            response = self.session.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def close(self):
        """Fecha a sessão HTTP"""
        self.session.close() 