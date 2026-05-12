from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient
from langchain_core.language_models.llms import LLM

from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)


class HuggingFaceInferenceLLM(LLM):
    repo_id: str
    hf_token: Optional[str] = None
    temperature: float = 0.3
    max_new_tokens: int = 256

    @property
    def _llm_type(self) -> str:
        return "huggingface_inference_client"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        client = InferenceClient(model=self.repo_id, token=self.hf_token)

        try:
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                stop=stop,
            )
            return response.choices[0].message.content.strip()
        except Exception as chat_error:
            logger.warning("Chat completion failed, falling back to text generation: %s", chat_error)
            return client.text_generation(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                return_full_text=False,
                stop=stop,
            ).strip()


def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info("Loading LLM from HuggingFace")

        llm = HuggingFaceInferenceLLM(
            repo_id=huggingface_repo_id,
            hf_token=hf_token,
            temperature=0.3,
            max_new_tokens=256,
        )

        logger.info("LLM loaded successfully")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load a LLM", e)
        logger.error(str(error_message))
