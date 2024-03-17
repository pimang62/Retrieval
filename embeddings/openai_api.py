# OpenAI model for embedding
# https://github.com/hwchase17/langchain/blob/5f17c57174c88e8c00bd71216dcf44b14fee7aaf/langchain/embeddings/openai.py#L25
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Callable, Union

from pydantic import model_validator, BaseModel, Field, ConfigDict
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import sys
sys.path.append('../../')
from .base import Embeddings
from utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


def _create_retry_decorator(
    embeddings: OpenAIEmbedding
) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10

    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)
            | retry_if_exception_type(openai.error.ServiceUnavailableError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

def _check_response(response: dict):
    if any(len(d["embedding"]) == 1 for d in response["data"]):
        import openai

        raise openai.error.APIError("OpenAI API returned an empty embedding")
    return response

def embed_with_retry(embeddings: OpenAIEmbedding, **kwargs: Any) -> Any:
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        response = embeddings.client.create(**kwargs)
        return _check_response(response)

    return _embed_with_retry(**kwargs)

class OpenAIEmbedding(BaseModel, Embeddings):
    client: Any = None
    tokenizer: Any = None
    model: str = "text-embedding-ada-002" # "text-embedding-3-large"
    deployment: str = None
    openai_api_version: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_key: Optional[str] = None
    embedding_ctx_length: int = 8191
    max_retries: int = 3
    #headers: Any = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        #all_required_filed_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            #if field_name not in all_required_filed_name:
            #    warnings.warn(
            #        f"""WARNING! {field_name} is not default parameter.
            #        {filed_name} was transferred to model_kwargs.
            #        Please confirm that {field_name} is what you intended."""
            #    )
            #    extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in env."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values, "openai_api_base", "OPENAI_API_BASE", default="",
        )
        values["openai_api_type"] = get_from_dict_or_env(
            values, "openai_api_type", "OPENAI_API_TYPE", default="",
        )
        values["openai_api_version"] = get_from_dict_or_env(
            values, "openai_api_version", "OPENAI_API_VERSION", default="",
        )
        values["deployment"] = get_from_dict_or_env(
            values, "deployment", "EMBEDDING_DEPLOYMENT", default="",
        )
        try:
            import openai

            values["client"] = openai.Embedding
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _invocation_params(self) -> Dict:
        openai_args = {
            "model": self.model,
            #"request_timeout": self.request_timeout,
            #"headers": self.headers,
            "api_key": self.openai_api_key,
            "api_base": self.openai_api_base,
            "api_type": self.openai_api_type,
            "api_version": self.openai_api_version,
            "deployment_id": self.deployment,
            **self.model_kwargs,
        }
        return openai_args
    
    def _get_len_safe_embeddings(
        self, texts: List[str], *, engine: str, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens += [token[j : j + self.embedding_ctx_length]]
                indices += [i]

        batched_embeddings = []
        _chunk_size = chunk_size or self.chunk_size

        if self.show_progress_bar:
            try:
                import tqdm

                _iter = tqdm.tqdm(range(0, len(tokens), _chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), _chunk_size)
        else:
            _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            response = embed_with_retry(
                self,
                input=tokens[i : i + _chunk_size],
                **self._invocation_params,
            )
            batched_embeddings += [r["embedding"] for r in response["data"]]

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = embed_with_retry(
                    self,
                    input="",
                    **self._invocation_params,
                )[
                    "data"
                ][0]["embedding"]
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings
    
    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint."""
        # printing which model you use
        # handle large input text
        if len(text) > self.embedding_ctx_length:
            return self._get_len_safe_embeddings([text], engine=engine)[0]
        else:
            if self.model.endswith("001"):
                # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
                # replace newlines, which can negatively affect performance.
                text = text.replace("\n", " ")
            return embed_with_retry(
                self,
                input=[text],
                **self._invocation_params,
            )[
                "data"
            ][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute document embeddings"""
        print("Loading openai embedding model", self.model)
        embeddings = []
        for text in texts:
            embeddings.append(self._embedding_func(text, engine=self.model))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings"""
        embedding = self._embedding_func(text, engine=self.model)
        return embedding