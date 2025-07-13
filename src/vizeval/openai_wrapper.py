"""
Wrapper transparente para OpenAI com integração automática da avaliação Vizeval
"""

import copy
import logging
from typing import Any, Dict, Optional, List, Union
from openai import OpenAI as _OpenAI
from openai.types.chat import ChatCompletion

from .client import VizevalClient
from .models import VizevalConfig, VizevalResult, RetryAttempt, EvaluationResponse
from .exceptions import VizevalOpenAIError, VizevalThresholdError, VizevalConfigError
from .evaluators import validate_evaluator

# Configurar logging
logger = logging.getLogger(__name__)


class OpenAI(_OpenAI):
    """
    Wrapper transparente para OpenAI que integra automaticamente com Vizeval
    
    Substitui a classe OpenAI padrão e intercepta chamadas para chat.completions.create
    para avaliar automaticamente as respostas e implementar retry baseado em threshold.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        vizeval_config: Optional[Union[VizevalConfig, Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Inicializa o wrapper OpenAI com configuração Vizeval
        
        Args:
            api_key: Chave de API do OpenAI
            vizeval_config: Configuração Vizeval ou dict com configurações
            **kwargs: Argumentos passados para o OpenAI original
        """
        super().__init__(api_key=api_key, **kwargs)
        
        # Configurar Vizeval
        if isinstance(vizeval_config, dict):
            self.vizeval_config = VizevalConfig(**vizeval_config)
        elif isinstance(vizeval_config, VizevalConfig):
            self.vizeval_config = vizeval_config
        else:
            self.vizeval_config = None
        
        # Inicializar cliente Vizeval se configurado
        if self.vizeval_config:
            self.vizeval_client = VizevalClient(
                api_key=self.vizeval_config.api_key,
                base_url=self.vizeval_config.base_url
            )
        else:
            self.vizeval_client = None
    
    def set_vizeval_config(self, config: Union[VizevalConfig, Dict[str, Any]]):
        """
        Define ou atualiza a configuração Vizeval
        
        Args:
            config: Configuração Vizeval ou dict
        """
        if isinstance(config, dict):
            self.vizeval_config = VizevalConfig(**config)
        elif isinstance(config, VizevalConfig):
            self.vizeval_config = config
        else:
            raise VizevalConfigError("config deve ser VizevalConfig ou dict")
        
        # Recriar cliente
        self.vizeval_client = VizevalClient(
            api_key=self.vizeval_config.api_key,
            base_url=self.vizeval_config.base_url
        )
    
    def disable_vizeval(self):
        """Desativa a integração Vizeval"""
        self.vizeval_config = None
        self.vizeval_client = None
    
    @property
    def chat(self):
        """Retorna o wrapper de chat que intercepta completions"""
        return ChatWrapper(self)


class ChatWrapper:
    """Wrapper para interceptar chamadas de chat.completions"""
    
    def __init__(self, openai_wrapper: OpenAI):
        self.openai_wrapper = openai_wrapper
        self.original_chat = super(OpenAI, openai_wrapper).chat
    
    @property
    def completions(self):
        """Retorna o wrapper de completions"""
        return CompletionsWrapper(self.openai_wrapper, self.original_chat.completions)


class CompletionsWrapper:
    """Wrapper para interceptar chamadas de completions.create"""
    
    def __init__(self, openai_wrapper: OpenAI, original_completions):
        self.openai_wrapper = openai_wrapper
        self.original_completions = original_completions
    
    def create(self, **kwargs) -> Union[ChatCompletion, VizevalResult]:
        """
        Intercepta chat.completions.create para integrar com Vizeval
        
        Args:
            **kwargs: Argumentos para chat.completions.create
            
        Returns:
            ChatCompletion ou VizevalResult dependendo da configuração
        """
        # Se Vizeval não estiver configurado, usar OpenAI normalmente
        if not self.openai_wrapper.vizeval_config:
            return self.original_completions.create(**kwargs)
        
        # Verificar se é uma chamada válida para avaliação
        if not self._is_evaluable_call(kwargs):
            logger.warning("Chamada não é avaliável pelo Vizeval, usando OpenAI normalmente")
            return self.original_completions.create(**kwargs)
        
        # Executar com retry automático
        return self._create_with_vizeval_retry(kwargs)
    
    def _is_evaluable_call(self, kwargs: Dict[str, Any]) -> bool:
        """
        Verifica se a chamada pode ser avaliada pelo Vizeval
        
        Args:
            kwargs: Argumentos da chamada
            
        Returns:
            True se pode ser avaliada
        """
        # Verificar se tem messages
        if "messages" not in kwargs:
            return False
        
        messages = kwargs["messages"]
        if not messages or not isinstance(messages, list):
            return False
        
        # Verificar se tem pelo menos uma mensagem do usuário
        user_messages = [m for m in messages if m.get("role") == "user"]
        return len(user_messages) > 0
    
    def _create_with_vizeval_retry(self, kwargs: Dict[str, Any]) -> VizevalResult:
        """
        Executa a chamada OpenAI com retry automático baseado no Vizeval
        
        Args:
            kwargs: Argumentos da chamada
            
        Returns:
            VizevalResult com resultado final e histórico
        """
        config = self.openai_wrapper.vizeval_config
        attempts = []
        best_response = None
        best_evaluation = None
        best_score = 0.0
        
        for attempt in range(config.max_retries + 1):
            try:
                # Fazer chamada OpenAI
                response = self.original_completions.create(**kwargs)
                
                # Extrair informações para avaliação
                system_prompt, user_prompt = self._extract_prompts(kwargs["messages"])
                llm_response = self._extract_response_content(response)
                
                # Avaliar com Vizeval
                evaluation = self.openai_wrapper.vizeval_client.evaluate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response=llm_response,
                    evaluator=config.evaluator,
                    metadata=config.metadata,
                    async_mode=config.async_mode
                )
                
                # Registrar tentativa
                retry_attempt = RetryAttempt(
                    attempt_number=attempt + 1,
                    score=evaluation.score,
                    feedback=evaluation.feedback,
                    openai_response=response,
                    evaluation_response=evaluation
                )
                attempts.append(retry_attempt)
                
                # Verificar se passou do threshold
                if evaluation.score is not None and evaluation.score >= config.threshold:
                    logger.info(f"Threshold atingido na tentativa {attempt + 1}: {evaluation.score}")
                    return VizevalResult(
                        final_response=response,
                        final_evaluation=evaluation,
                        attempts=attempts,
                        config=config
                    )
                
                # Manter melhor resultado
                if evaluation.score is not None and evaluation.score > best_score:
                    best_score = evaluation.score
                    best_response = response
                    best_evaluation = evaluation
                
                # Se não é a última tentativa, acrescentar contexto e ajustar parâmetros para retry
                if attempt < config.max_retries:
                    # 1) Mensagem do assistente com a resposta reprovada
                    assistant_msg = {
                        "role": "assistant",
                        "content": llm_response
                    }

                    # 2) Mensagem de crítica/feedback proveniente da avaliação Vizeval
                    score_str = f"{evaluation.score:.3f}" if evaluation.score is not None else "N/A"

                    critique_content = (
                        "Sua última resposta foi reprovada pela avaliação de qualidade médica."
                        "Considerando o score {score_str}, reescreva a resposta anterior para melhorar a qualidade médica."
                    )

                    critique_msg = {
                        "role": "system",
                        "content": critique_content
                    }

                    # Construir novo array de mensagens incluindo histórico e feedback
                    new_kwargs = copy.deepcopy(kwargs)
                    new_kwargs["messages"] = kwargs["messages"] + [assistant_msg, critique_msg]

                    # Ajustar parâmetros (temperature, top_p, etc.) e seguir para próximo retry
                    kwargs = self._adjust_parameters_for_retry(new_kwargs, attempt + 1)
                    logger.info(
                        f"Tentativa {attempt + 1} não passou do threshold ({evaluation.score}), tentando novamente com contexto adicional..."
                    )
                
            except Exception as e:
                logger.error(f"Erro na tentativa {attempt + 1}: {str(e)}")
                # Se der erro, usar a melhor resposta obtida até agora
                if best_response is not None:
                    break
                # Se não tem nenhuma resposta válida, propagar erro
                if attempt == config.max_retries:
                    raise VizevalOpenAIError(f"Todas as tentativas falharam: {str(e)}")
        
        # Se chegou aqui, não conseguiu atingir o threshold
        if best_response is None:
            raise VizevalOpenAIError("Não foi possível obter nenhuma resposta válida")
        
        logger.warning(f"Threshold não atingido após {config.max_retries + 1} tentativas. Melhor score: {best_score}")
        
        return VizevalResult(
            final_response=best_response,
            final_evaluation=best_evaluation,
            attempts=attempts,
            config=config
        )
    
    def _extract_prompts(self, messages: List[Dict[str, Any]]) -> tuple[str, str]:
        """
        Extrai system_prompt e user_prompt das mensagens
        
        Args:
            messages: Lista de mensagens
            
        Returns:
            Tuple com (system_prompt, user_prompt)
        """
        system_prompt = ""
        user_prompt = ""
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                # Concatenar mensagens do usuário
                user_prompt += content + "\n"
        
        return system_prompt.strip(), user_prompt.strip()
    
    def _extract_response_content(self, response: ChatCompletion) -> str:
        """
        Extrai o conteúdo da resposta do OpenAI
        
        Args:
            response: Resposta do OpenAI
            
        Returns:
            Conteúdo da resposta
        """
        try:
            return response.choices[0].message.content or ""
        except (IndexError, AttributeError):
            return ""
    
    def _adjust_parameters_for_retry(self, kwargs: Dict[str, Any], attempt: int) -> Dict[str, Any]:
        """
        Ajusta parâmetros para retry (pode aumentar temperature, etc.)
        
        Args:
            kwargs: Argumentos originais
            attempt: Número da tentativa
            
        Returns:
            Novos argumentos ajustados
        """
        new_kwargs = copy.deepcopy(kwargs)
        
        # Ajustar temperature gradualmente
        current_temp = new_kwargs.get("temperature", 0.7)
        
        # Aumentar temperature em pequenos incrementos
        if current_temp < 0.9:
            new_kwargs["temperature"] = min(0.9, current_temp + 0.1)
        
        # Ajustar top_p se necessário
        if "top_p" in new_kwargs:
            current_top_p = new_kwargs["top_p"]
            if current_top_p < 0.95:
                new_kwargs["top_p"] = min(0.95, current_top_p + 0.05)
        
        return new_kwargs 