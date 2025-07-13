"""
Testes básicos para a SDK Vizeval
"""

import pytest
from unittest.mock import Mock, patch
from vizeval import VizevalClient, VizevalConfig, Evaluator
from vizeval.models import EvaluationRequest, EvaluationResponse
from vizeval.exceptions import VizevalConfigError, VizevalAPIError


class TestVizevalConfig:
    """Testes para VizevalConfig"""
    
    def test_config_creation(self):
        """Testa criação de configuração básica"""
        config = VizevalConfig(
            api_key="test_key",
            evaluator="medical",
            threshold=0.8,
            max_retries=3
        )
        
        assert config.api_key == "test_key"
        assert config.evaluator == "medical"
        assert config.threshold == 0.8
        assert config.max_retries == 3
    
    def test_config_validation(self):
        """Testa validação de configuração"""
        # Threshold inválido
        with pytest.raises(ValueError, match="threshold deve estar entre 0 e 1"):
            VizevalConfig(api_key="test", threshold=1.5)
        
        # Max retries inválido
        with pytest.raises(ValueError, match="max_retries deve ser maior ou igual a 0"):
            VizevalConfig(api_key="test", max_retries=-1)


class TestVizevalClient:
    """Testes para VizevalClient"""
    
    def test_client_creation(self):
        """Testa criação do cliente"""
        client = VizevalClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.base_url == "https://api.vizeval.com"
    
    def test_client_creation_without_api_key(self):
        """Testa criação do cliente sem API key"""
        with pytest.raises(VizevalConfigError, match="api_key é obrigatório"):
            VizevalClient(api_key="")
    
    @patch('vizeval.client.requests.Session.post')
    def test_evaluate_success(self, mock_post):
        """Testa avaliação bem-sucedida"""
        # Mock da resposta
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "evaluator": "medical",
            "score": 0.85,
            "feedback": "Resposta adequada"
        }
        mock_post.return_value = mock_response
        
        client = VizevalClient(api_key="test_key")
        result = client.evaluate(
            system_prompt="Você é um médico",
            user_prompt="Quais são os sintomas da gripe?",
            response="Febre, dor de cabeça, fadiga",
            evaluator="medical"
        )
        
        assert result.score == 0.85
        assert result.feedback == "Resposta adequada"
        assert result.evaluator == "medical"
    
    @patch('vizeval.client.requests.Session.post')
    def test_evaluate_api_error(self, mock_post):
        """Testa erro na API"""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Erro de validação"}
        mock_post.return_value = mock_response
        
        client = VizevalClient(api_key="test_key")
        
        with pytest.raises(VizevalAPIError):
            client.evaluate(
                system_prompt="test",
                user_prompt="test",
                response="test",
                evaluator="medical"
            )
    
    def test_evaluate_invalid_evaluator(self):
        """Testa avaliação com evaluator inválido"""
        client = VizevalClient(api_key="test_key")
        
        with pytest.raises(VizevalConfigError, match="Evaluator 'invalid' não é válido"):
            client.evaluate(
                system_prompt="test",
                user_prompt="test",
                response="test",
                evaluator="invalid"
            )


class TestEvaluationModels:
    """Testes para modelos de avaliação"""
    
    def test_evaluation_request_creation(self):
        """Testa criação de requisição de avaliação"""
        request = EvaluationRequest(
            system_prompt="Você é um médico",
            user_prompt="Quais são os sintomas?",
            response="Febre, dor de cabeça",
            evaluator="medical",
            api_key="test_key"
        )
        
        assert request.system_prompt == "Você é um médico"
        assert request.user_prompt == "Quais são os sintomas?"
        assert request.response == "Febre, dor de cabeça"
        assert request.evaluator == "medical"
        assert request.api_key == "test_key"
    
    def test_evaluation_response_properties(self):
        """Testa propriedades da resposta de avaliação"""
        # Resposta com score
        response = EvaluationResponse(
            evaluator="medical",
            score=0.85,
            feedback="Boa resposta"
        )
        
        assert response.is_success is True
        assert response.passed_threshold(0.8) is True
        assert response.passed_threshold(0.9) is False
        
        # Resposta sem score
        response_no_score = EvaluationResponse(
            evaluator="medical",
            score=None,
            feedback="Erro na avaliação"
        )
        
        assert response_no_score.is_success is False
        assert response_no_score.passed_threshold(0.8) is False


class TestEvaluators:
    """Testes para evaluators"""
    
    def test_evaluator_enum(self):
        """Testa enum de evaluators"""
        assert Evaluator.MEDICAL == "medical"
        assert Evaluator.JURIDICAL == "juridical"
        assert Evaluator.DUMMY == "dummy"
    
    def test_evaluator_validation(self):
        """Testa validação de evaluators"""
        from vizeval.evaluators import validate_evaluator
        
        assert validate_evaluator("medical") is True
        assert validate_evaluator("juridical") is True
        assert validate_evaluator("dummy") is True
        assert validate_evaluator("invalid") is False
    
    def test_evaluator_info(self):
        """Testa informações dos evaluators"""
        from vizeval.evaluators import get_evaluator_info
        
        info = get_evaluator_info("medical")
        assert info["threshold"] == 0.8
        assert "médico" in info["description"]
        
        with pytest.raises(ValueError, match="Evaluator 'invalid' não disponível"):
            get_evaluator_info("invalid")


if __name__ == "__main__":
    pytest.main([__file__]) 