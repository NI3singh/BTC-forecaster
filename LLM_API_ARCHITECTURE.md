# LLM API Architecture — System Design Spec

> A provider-agnostic LLM client layer. Hand this document to an AI/engineer to
> replicate the same architecture in another project. It is faithful to the
> reference implementation and includes a clean, generalized skeleton at the end.

---

## 0. Mental model (read this first)

A **config-driven factory** returns ready-to-use LangChain chat-model objects. Agents
never construct an LLM themselves and never branch on provider — they call
`create_llm_client(provider, model, …).get_llm()` and get back something with a
uniform `.invoke()` / `.with_structured_output()` interface.

Variation is split along **two axes**, handled in two different places so neither
becomes an `if`-ladder:

| Axis | Example | Where it lives |
|---|---|---|
| Genuinely different SDK/API | Anthropic, Google, Azure, Bedrock | A dedicated **client class** per provider |
| Same wire format (OpenAI Chat Completions), different URL/key/quirk | xAI, DeepSeek, Qwen, GLM, MiniMax, Groq, Ollama, vLLM… | **One declarative registry row** (`ProviderSpec`) |
| Per-*model* API quirks (rejects `tool_choice`, needs `reasoning_split`…) | DeepSeek-reasoner, MiniMax-M2.x | A **capability table** keyed by model id |

The guiding rule: **adding a provider or model = adding a data row, not editing logic.**

---

## 1. Design principles

1. **Single source of truth per concern.** One map for provider→API-key env var; one
   registry for OpenAI-compatible providers; one capability table for model quirks; one
   catalog for model lists. Nothing is duplicated.
2. **Declarative over imperative.** Behavior is expressed as frozen dataclass rows in
   dicts, resolved at runtime — not as branches in client code.
3. **Lazy imports.** The factory imports a provider's SDK only when that provider is
   requested, so importing the package (e.g. in tests) never pulls heavy SDKs or fails on
   missing keys. The optional Bedrock dependency (`langchain-aws`) is imported on first use.
4. **Normalization layer.** Reasoning/Responses APIs return content as a *list of typed
   blocks*; a `normalize_content()` shim collapses it to a plain string so every downstream
   consumer sees the same shape.
5. **Graceful degradation.** Structured output is attempted; if the model/provider can't
   do it, the code falls back to free-text instead of crashing.
6. **Config overridable by env vars**, with type coercion driven by the default value's type.

---

## 2. Module layout

```
mypkg/
├─ __init__.py              # loads .env at import time (override=False)
├─ default_config.py        # DEFAULT_CONFIG dict + APP_*-style env overlay
└─ llm_clients/
   ├─ __init__.py           # public API: create_llm_client, BaseLLMClient
   ├─ factory.py            # create_llm_client(provider, model, base_url, **kw) -> client
   ├─ base_client.py        # BaseLLMClient ABC + normalize_content()
   ├─ api_key_env.py        # PROVIDER_API_KEY_ENV: provider -> env-var name
   ├─ openai_client.py      # ProviderSpec + OPENAI_COMPATIBLE_PROVIDERS registry + OpenAIClient
   ├─ anthropic_client.py   # native Anthropic client
   ├─ google_client.py      # native Gemini client
   ├─ azure_client.py       # Azure OpenAI client
   ├─ bedrock_client.py     # AWS Bedrock (lazy optional dep)
   ├─ capabilities.py       # ModelCapabilities table (per-model API quirks)
   ├─ validators.py         # validate_model(provider, model)
   └─ model_catalog.py      # MODEL_OPTIONS: CLI dropdowns + known-model lists
```

The **public surface is tiny** — only two names are exported:

```python
# llm_clients/__init__.py
from .base_client import BaseLLMClient
from .factory import create_llm_client
__all__ = ["BaseLLMClient", "create_llm_client"]
```

---

## 3. End-to-end data flow

```
.env  ──load_dotenv()──►  os.environ
                                │
default_config.DEFAULT_CONFIG ──┤ (env overlay w/ type coercion)
                                ▼
consumer (e.g. graph __init__):
    llm_kwargs = _get_provider_kwargs(config)   # reasoning_effort / thinking_level / effort / temperature
    client = create_llm_client(provider, model, base_url, **llm_kwargs)
                                │
                                ▼
              factory dispatch by provider name
            ┌───────────────┬───────────────────────────┐
       native APIs     OpenAI-compatible family     (raise if unknown)
   Anthropic/Google/    OpenAIClient + registry row
   Azure/Bedrock                │
                                ▼
                         client.get_llm()
                                │
                                ▼
            a LangChain Chat* instance (Normalized* subclass)
                                │
                ┌───────────────┴───────────────┐
          .invoke(prompt)            .with_structured_output(Schema)
        (content normalized        (method chosen from capability table)
         to a string)
```

---

## 4. Component-by-component spec

### 4.1 Env loading — `mypkg/__init__.py`

`.env` is loaded **at package import** so every consumer (config overlay + clients) sees
keys regardless of entry point. Crucially `override=False` so it never clobbers a value
the caller already exported, and `find_dotenv(usecwd=True)` walks up from the CWD (so an
installed console script finds the *project's* `.env`, not one next to site-packages).

```python
try:
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
    load_dotenv(find_dotenv(".env.enterprise", usecwd=True), override=False)  # optional 2nd file
except ImportError:
    pass
```

### 4.2 Config + env overlay — `default_config.py`

A single `DEFAULT_CONFIG` dict holds LLM settings (`llm_provider`, `deep_think_llm`,
`quick_think_llm`, `backend_url`, the per-provider thinking knobs, `temperature`). A
declarative `_ENV_OVERRIDES` map lets any `APP_*` env var replace a config key, **coerced
to the type of the existing default**:

```python
_ENV_OVERRIDES = {
    "APP_LLM_PROVIDER":   "llm_provider",
    "APP_DEEP_THINK_LLM": "deep_think_llm",
    "APP_QUICK_THINK_LLM":"quick_think_llm",
    "APP_LLM_BACKEND_URL":"backend_url",
    "APP_TEMPERATURE":    "temperature",
}

def _coerce(value: str, reference):
    if isinstance(reference, bool):
        return value.strip().lower() in ("true", "1", "yes", "on")
    if isinstance(reference, int) and not isinstance(reference, bool):
        return int(value)
    if isinstance(reference, float):
        return float(value)
    return value

def _apply_env_overrides(config: dict) -> dict:
    for env_var, key in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_var)
        if raw is None or raw == "":
            continue
        config[key] = _coerce(raw, config.get(key))
    return config
```

Relevant defaults (note `backend_url=None` so each provider uses its own endpoint; the
thinking knobs are all `None` = leave provider at its default):

```python
"llm_provider": "openai",
"deep_think_llm": "gpt-5.5",
"quick_think_llm": "gpt-5.4-mini",
"backend_url": None,
"google_thinking_level": None,    # "high"/"minimal"/...
"openai_reasoning_effort": None,  # "low"/"medium"/"high"
"anthropic_effort": None,         # "low"/"medium"/"high"
"temperature": None,              # cross-provider; None = provider default
```

### 4.3 Provider → API-key env var — `api_key_env.py`

The **one** place that knows which env var holds each provider's key. Consulted by both
the client (to read the key) and the CLI (to prompt for it). `None` means "no single key"
(Ollama = local; Bedrock = AWS credential chain).

```python
PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY", "azure": "AZURE_OPENAI_API_KEY",
    "bedrock": None,                       # AWS credential chain
    "xai": "XAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY",
    "ollama": None,                        # local, no auth
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
    # …dual-region providers each get their own key (intl vs CN not interchangeable)
}
def get_api_key_env(provider: str) -> str | None:
    return PROVIDER_API_KEY_ENV.get(provider.lower())
```

### 4.4 Base class + normalization — `base_client.py`

```python
def normalize_content(response):
    """Collapse list-of-typed-blocks content (Responses API, Gemini 3) to a string."""
    content = response.content
    if isinstance(content, list):
        texts = [
            item.get("text", "") if isinstance(item, dict) and item.get("type") == "text"
            else item if isinstance(item, str) else ""
            for item in content
        ]
        response.content = "\n".join(t for t in texts if t)
    return response

class BaseLLMClient(ABC):
    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs            # the passthrough bag — see §5

    def get_provider_name(self) -> str: ...
    def warn_if_unknown_model(self) -> None:   # warns (doesn't fail) on unknown model id
        if not self.validate_model():
            warnings.warn(..., RuntimeWarning)

    @abstractmethod
    def get_llm(self) -> Any: ...
    @abstractmethod
    def validate_model(self) -> bool: ...
```

### 4.5 Factory — `factory.py`

Dispatch order matters: **native APIs are matched first** (string check before importing
the OpenAI client), everything else falls through to the OpenAI-compatible registry. Each
branch imports lazily.

```python
def create_llm_client(provider, model, base_url=None, **kwargs) -> BaseLLMClient:
    p = provider.lower()
    if p == "anthropic":
        from .anthropic_client import AnthropicClient;  return AnthropicClient(model, base_url, **kwargs)
    if p == "google":
        from .google_client import GoogleClient;        return GoogleClient(model, base_url, **kwargs)
    if p == "azure":
        from .azure_client import AzureOpenAIClient;     return AzureOpenAIClient(model, base_url, **kwargs)
    if p == "bedrock":
        from .bedrock_client import BedrockClient;       return BedrockClient(model, base_url, **kwargs)
    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(p):
        return OpenAIClient(model, base_url, provider=p, **kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider}")
```

### 4.6 The OpenAI-compatible registry — `openai_client.py` (the heart of the design)

A frozen dataclass describes one provider declaratively; a dict is the single source of
truth for the whole family.

```python
@dataclass(frozen=True)
class ProviderSpec:
    chat_class: type = NormalizedChatOpenAI  # subclass carrying provider quirks
    base_url: str | None = None              # default endpoint (None -> SDK default)
    base_url_env: str | None = None          # env var that overrides base_url (e.g. OLLAMA_BASE_URL)
    key_optional: bool = False               # don't require/prompt; send placeholder if unset
    placeholder_key: str = "EMPTY"           # sent when keyless (local servers)
    require_base_url: bool = False           # error if no base_url resolved (generic endpoint)
    use_responses_api: bool = False          # native OpenAI Responses API

OPENAI_COMPATIBLE_PROVIDERS: dict[str, ProviderSpec] = {
    "openai":     ProviderSpec(use_responses_api=True),
    "xai":        ProviderSpec(base_url="https://api.x.ai/v1"),
    "deepseek":   ProviderSpec(base_url="https://api.deepseek.com", chat_class=DeepSeekChatOpenAI),
    "groq":       ProviderSpec(base_url="https://api.groq.com/openai/v1"),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1", base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama"),
    "openai_compatible": ProviderSpec(require_base_url=True, key_optional=True),  # vLLM/LM Studio/relay
    # …xai, qwen, glm, minimax, mistral, kimi, nvidia all just rows
}

def is_openai_compatible(provider: str) -> bool:
    return provider.lower() in OPENAI_COMPATIBLE_PROVIDERS
```

`OpenAIClient.get_llm()` resolves everything from the spec. **base_url precedence** is the
rule to copy exactly: explicit client `base_url` > provider `base_url_env` > spec default >
SDK default.

```python
def get_llm(self):
    self.warn_if_unknown_model()
    llm_kwargs = {"model": self.model}
    spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
    chat_cls = spec.chat_class if spec else NormalizedChatOpenAI

    if spec:
        env_base_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
        base_url = self.base_url or env_base_url or spec.base_url
        if spec.require_base_url and not base_url:
            raise ValueError(f"Provider '{self.provider}' requires a base_url …")
        if base_url:
            llm_kwargs["base_url"] = base_url

        api_key_env = get_api_key_env(self.provider)
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key:
            llm_kwargs["api_key"] = api_key
        elif spec.key_optional:
            llm_kwargs["api_key"] = spec.placeholder_key      # keyless local servers
        elif api_key_env:
            raise ValueError(f"API key for '{self.provider}' not set. Set {api_key_env} …")

        # Responses API exists ONLY on native OpenAI — a custom base_url speaks only Chat Completions
        if spec.use_responses_api and _is_native_openai_base_url(base_url):
            llm_kwargs["use_responses_api"] = True

    for key in _PASSTHROUGH_KWARGS:          # see §5
        if key in self.kwargs:
            llm_kwargs[key] = self.kwargs[key]
    return chat_cls(**llm_kwargs)
```

**Provider quirks live in `ChatOpenAI` subclasses, not in the client.** Two patterns:

- `NormalizedChatOpenAI` — overrides `invoke()` (normalize content) and
  `with_structured_output()` (consults the capability table; suppresses `tool_choice` when
  the model rejects it).
- `DeepSeekChatOpenAI` / `MinimaxChatOpenAI` — override the LangChain request/response
  hooks (`_get_request_payload`, `_create_chat_result`) for wire-format quirks (DeepSeek's
  `reasoning_content` round-trip; MiniMax's `reasoning_split` via `extra_body`). These are
  *only* reached when the registry row points `chat_class` at them.

### 4.7 Per-model capability table — `capabilities.py`

The single place that knows which *model id* rejects which API param. The
structured-output dispatch reads this instead of hardcoding model names.

```python
@dataclass(frozen=True)
class ModelCapabilities:
    supports_tool_choice: bool
    supports_json_mode: bool
    supports_json_schema: bool
    preferred_structured_method: Literal["function_calling","json_mode","json_schema","none"]
    requires_reasoning_content_roundtrip: bool = False
    requires_reasoning_split: bool = False

_DEFAULT = ModelCapabilities(True, True, True, "function_calling")
_BY_ID: dict[str, ModelCapabilities] = { "deepseek-reasoner": _DEEPSEEK_THINKING, … }
_BY_PATTERN = [(re.compile(r"^deepseek-v\d"), _DEEPSEEK_THINKING), …]  # forward-compat

def get_capabilities(model_name: str) -> ModelCapabilities:
    if model_name in _BY_ID: return _BY_ID[model_name]          # exact id wins
    for pat, caps in _BY_PATTERN:
        if pat.match(model_name): return caps                   # then pattern
    return _DEFAULT                                             # then default
```

Consumed inside `NormalizedChatOpenAI.with_structured_output`:

```python
def with_structured_output(self, schema, *, method=None, **kwargs):
    caps = get_capabilities(self.model_name)
    if caps.preferred_structured_method == "none":
        raise NotImplementedError(...)        # caller falls back to free text
    method = method or caps.preferred_structured_method
    if method == "function_calling" and not caps.supports_tool_choice:
        kwargs.setdefault("tool_choice", None)   # bind schema as a tool, but send no tool_choice
    return super().with_structured_output(schema, method=method, **kwargs)
```

### 4.8 Validation + catalog — `validators.py`, `model_catalog.py`

- `model_catalog.MODEL_OPTIONS` is `{provider: {"quick": [(label, id)…], "deep": […]}}` —
  drives CLI dropdowns. `get_known_models()` derives the flat known-id set from it.
- `validators.validate_model(provider, model)` returns `True` for "any-model" providers
  (Ollama, OpenRouter, generic endpoint, Bedrock, hosted relays) and for unknown providers;
  otherwise checks membership. It only ever drives a **warning**, never a hard failure —
  unknown models still run.

### 4.9 Native-API clients (same shape, different SDK)

Each is ~40 lines: subclass the LangChain chat class to normalize content, then
`get_llm()` assembles kwargs from a provider-specific passthrough tuple. Highlights:

- **Anthropic** — `effort` (extended thinking) is gated: only sent for models matching
  `^claude-(opus|sonnet)-\d+-\d+$` (Haiku 400s on it). Forward-compatible by regex.
- **Google** — maps a unified `thinking_level` to the right param per model family
  (`thinking_level` for Gemini 3, `thinking_budget` for 2.5); maps unified `api_key` →
  `google_api_key`.
- **Azure** — reads `AZURE_OPENAI_*` env (endpoint, deployment, api version);
  `validate_model()` returns `True` (any deployed name).
- **Bedrock** — lazy-imports the optional `langchain-aws`; auth via AWS credential chain +
  region resolution (`AWS_REGION` → `AWS_DEFAULT_REGION` → default).

### 4.10 Structured-output helper — `agents/utils/structured.py`

The canonical two-step pattern every agent uses, with a fallback so the pipeline never blocks:

```python
def bind_structured(llm, schema, agent_name):
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning("%s: no structured output (%s); using free text", agent_name, exc)
        return None

def invoke_structured_or_freetext(structured_llm, plain_llm, prompt, render, agent_name):
    if structured_llm is not None:
        try:
            return render(structured_llm.invoke(prompt))   # typed Pydantic -> markdown
        except Exception as exc:
            logger.warning("%s: structured call failed (%s); retrying as free text", agent_name, exc)
    return plain_llm.invoke(prompt).content
```

### 4.11 Consumer wiring — how config becomes provider kwargs

The consumer translates **config keys → provider-specific kwargs** in one small method,
then builds both a "deep" and "quick" LLM through the same factory:

```python
def _get_provider_kwargs(self) -> dict:
    kwargs = {}
    p = self.config.get("llm_provider", "").lower()
    if p == "google"   and self.config.get("google_thinking_level"):
        kwargs["thinking_level"]   = self.config["google_thinking_level"]
    elif p == "openai" and self.config.get("openai_reasoning_effort"):
        kwargs["reasoning_effort"] = self.config["openai_reasoning_effort"]
    elif p == "anthropic" and self.config.get("anthropic_effort"):
        kwargs["effort"]           = self.config["anthropic_effort"]
    t = self.config.get("temperature")            # cross-provider
    if t is not None and t != "":
        kwargs["temperature"] = float(t)
    return kwargs

# build:
llm_kwargs = self._get_provider_kwargs()
if self.callbacks: llm_kwargs["callbacks"] = self.callbacks
self.deep_llm  = create_llm_client(provider, config["deep_think_llm"],  config.get("backend_url"), **llm_kwargs).get_llm()
self.quick_llm = create_llm_client(provider, config["quick_think_llm"], config.get("backend_url"), **llm_kwargs).get_llm()
```

### 4.12 CLI key prompting — `cli/utils.ensure_api_key`

Resolves a provider's key: returns it if already in env; for key-optional providers
reads-but-never-prompts; otherwise prompts (hidden input), **persists to `.env` via
`dotenv.set_key`**, and exports into `os.environ` for the running process. Returns `None`
for keyless providers. It reuses `get_api_key_env()` and the registry's `key_optional`
flag — no separate knowledge of providers.

---

## 5. The passthrough-kwargs pattern (important for correctness)

Each client declares a **tuple of kwarg names it forwards** to the LangChain constructor,
and copies only those that are present in `self.kwargs`. This is how arbitrary caller
options (`temperature`, `timeout`, `max_retries`, `callbacks`, `api_key`,
`reasoning_effort`/`effort`, `http_client`…) flow through without the base class knowing
about any of them.

```python
_PASSTHROUGH_KWARGS = ("timeout", "max_retries", "reasoning_effort", "temperature",
                       "api_key", "callbacks", "http_client", "http_async_client")
for key in _PASSTHROUGH_KWARGS:
    if key in self.kwargs:
        llm_kwargs[key] = self.kwargs[key]
```

The tuple differs per provider (Anthropic uses `effort`/`max_tokens` instead of
`reasoning_effort`; Google forwards `thinking_level` specially). This keeps each provider's
allowed surface explicit and prevents passing a param the SDK would reject.

---

## 6. Dependencies

```
langchain-core          # message types, base chat model
langchain-openai        # ChatOpenAI, AzureChatOpenAI
langchain-anthropic     # ChatAnthropic
langchain-google-genai  # ChatGoogleGenerativeAI
langchain-aws           # ChatBedrockConverse  (OPTIONAL extra — lazy import)
python-dotenv           # .env loading + set_key
pydantic                # structured-output schemas
```

Only `langchain-core` + whichever provider package you actually use are mandatory; the
lazy factory means an unused provider's SDK never has to be installed.

---

## 7. Extension recipes (the payoff of the design)

- **Add an OpenAI-compatible provider** → one row in `OPENAI_COMPATIBLE_PROVIDERS` + one
  row in `PROVIDER_API_KEY_ENV` (+ optional catalog entry). Zero logic changes.
- **Add a provider with a genuinely different API** → new `*_client.py` (subclass
  `BaseLLMClient`, normalize content) + one `if` branch in the factory + one
  `PROVIDER_API_KEY_ENV` row.
- **Handle a model-specific API quirk** → one `ModelCapabilities` row in
  `capabilities._BY_ID` (or a forward-compat regex in `_BY_PATTERN`). If it's a wire-format
  quirk, put it in a `ChatOpenAI` subclass and point the provider's `chat_class` at it.
- **Expose a new config value as an env override** → one row in `_ENV_OVERRIDES`.

---

## 8. Gotchas to preserve (the "without error" part)

1. **base_url precedence** must be `explicit > base_url_env > spec.base_url > SDK default`.
   Getting this wrong leaks one provider's endpoint into another.
2. **Responses API only on native OpenAI.** Gate `use_responses_api` behind a host check
   (`api.openai.com`/`*.openai.com`); a custom base_url on the `openai` provider speaks
   only Chat Completions.
3. **Keyless local servers** need a *placeholder* key (`"EMPTY"`/`"ollama"`) — the OpenAI
   SDK refuses to construct without some api_key string.
4. **Normalize content for every provider.** Reasoning/Responses/Gemini-3 return
   block-lists; without normalization, `.content` is a list and string ops downstream break.
5. **`tool_choice` suppression**: when a model rejects it, still bind the schema as a tool
   but `setdefault("tool_choice", None)` — don't drop the tool.
6. **`override=False` on dotenv** so a key exported in the real environment wins over the
   `.env` file.
7. **Validation warns, never fails** — unknown model ids must still run (new models ship
   faster than catalogs update).
8. **Lazy imports in the factory** — don't import all SDKs at module top, or `import` of
   the package fails wherever an optional SDK/key is missing.
9. **`effort`/thinking params are model-gated** — sending `effort` to Anthropic Haiku, or
   a `tool_choice` dict to MiniMax M2.x, returns HTTP 400. Gate by model id/regex.

---

## 9. Minimal generalized skeleton

A trimmed, provider-agnostic version of the same architecture you can drop into any
project and grow. It keeps every structural decision above; just fewer providers.

```python
# llm/base.py
import warnings
from abc import ABC, abstractmethod
from typing import Any

def normalize_content(response):
    c = response.content
    if isinstance(c, list):
        response.content = "\n".join(
            (i.get("text", "") if isinstance(i, dict) and i.get("type") == "text"
             else i if isinstance(i, str) else "")
            for i in c
        )
    return response

class BaseLLMClient(ABC):
    def __init__(self, model: str, base_url: str | None = None, **kwargs):
        self.model, self.base_url, self.kwargs = model, base_url, kwargs
    @abstractmethod
    def get_llm(self) -> Any: ...
    @abstractmethod
    def validate_model(self) -> bool: ...
```

```python
# llm/api_key_env.py
PROVIDER_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama": None,
    "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
}
def get_api_key_env(provider: str) -> str | None:
    return PROVIDER_API_KEY_ENV.get(provider.lower())
```

```python
# llm/openai_client.py
import os
from dataclasses import dataclass
from urllib.parse import urlparse
from langchain_openai import ChatOpenAI
from .base import BaseLLMClient, normalize_content
from .api_key_env import get_api_key_env

class NormalizedChatOpenAI(ChatOpenAI):
    def invoke(self, input, config=None, **kw):
        return normalize_content(super().invoke(input, config, **kw))

@dataclass(frozen=True)
class ProviderSpec:
    chat_class: type = NormalizedChatOpenAI
    base_url: str | None = None
    base_url_env: str | None = None
    key_optional: bool = False
    placeholder_key: str = "EMPTY"
    require_base_url: bool = False
    use_responses_api: bool = False

OPENAI_COMPATIBLE_PROVIDERS = {
    "openai":     ProviderSpec(use_responses_api=True),
    "openrouter": ProviderSpec(base_url="https://openrouter.ai/api/v1"),
    "ollama":     ProviderSpec(base_url="http://localhost:11434/v1",
                               base_url_env="OLLAMA_BASE_URL",
                               key_optional=True, placeholder_key="ollama"),
    "openai_compatible": ProviderSpec(require_base_url=True, key_optional=True),
}
def is_openai_compatible(p: str) -> bool:
    return p.lower() in OPENAI_COMPATIBLE_PROVIDERS

def _is_native_openai(base_url: str | None) -> bool:
    if not base_url: return True
    if "://" not in base_url: base_url = "https://" + base_url
    host = urlparse(base_url).hostname or ""
    return host == "api.openai.com" or host.endswith(".openai.com")

_PASSTHROUGH = ("timeout", "max_retries", "reasoning_effort", "temperature",
                "api_key", "callbacks")

class OpenAIClient(BaseLLMClient):
    def __init__(self, model, base_url=None, provider="openai", **kw):
        super().__init__(model, base_url, **kw)
        self.provider = provider.lower()
    def validate_model(self) -> bool:
        return True
    def get_llm(self):
        spec = OPENAI_COMPATIBLE_PROVIDERS.get(self.provider)
        chat_cls = spec.chat_class if spec else NormalizedChatOpenAI
        kw = {"model": self.model}
        if spec:
            env_url = os.environ.get(spec.base_url_env) if spec.base_url_env else None
            base_url = self.base_url or env_url or spec.base_url
            if spec.require_base_url and not base_url:
                raise ValueError(f"Provider '{self.provider}' requires a base_url.")
            if base_url: kw["base_url"] = base_url
            key_env = get_api_key_env(self.provider)
            key = os.environ.get(key_env) if key_env else None
            if key:                 kw["api_key"] = key
            elif spec.key_optional: kw["api_key"] = spec.placeholder_key
            elif key_env:           raise ValueError(f"Set {key_env} in your .env")
            if spec.use_responses_api and _is_native_openai(base_url):
                kw["use_responses_api"] = True
        elif self.base_url:
            kw["base_url"] = self.base_url
        for k in _PASSTHROUGH:
            if k in self.kwargs: kw[k] = self.kwargs[k]
        return chat_cls(**kw)
```

```python
# llm/anthropic_client.py
from langchain_anthropic import ChatAnthropic
from .base import BaseLLMClient, normalize_content

class NormalizedChatAnthropic(ChatAnthropic):
    def invoke(self, input, config=None, **kw):
        return normalize_content(super().invoke(input, config, **kw))

_PASSTHROUGH = ("timeout", "max_retries", "api_key", "max_tokens", "temperature", "callbacks")

class AnthropicClient(BaseLLMClient):
    def validate_model(self) -> bool:
        return True
    def get_llm(self):
        kw = {"model": self.model}
        if self.base_url: kw["base_url"] = self.base_url
        for k in _PASSTHROUGH:
            if k in self.kwargs: kw[k] = self.kwargs[k]
        return NormalizedChatAnthropic(**kw)
```

```python
# llm/factory.py
from .base import BaseLLMClient

def create_llm_client(provider, model, base_url=None, **kwargs) -> BaseLLMClient:
    p = provider.lower()
    if p == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)
    from .openai_client import OpenAIClient, is_openai_compatible
    if is_openai_compatible(p):
        return OpenAIClient(model, base_url, provider=p, **kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider}")
```

```python
# usage
from llm.factory import create_llm_client
llm = create_llm_client("openai", "gpt-5.5", temperature=0).get_llm()
print(llm.invoke("hello").content)        # always a plain string
```

---

## 10. One-paragraph summary

A factory returns LangChain chat objects; native APIs get their own small client class, the
OpenAI-compatible majority collapse into one declarative `ProviderSpec` registry, per-model
quirks live in a capability table, and three single-source-of-truth maps
(`PROVIDER_API_KEY_ENV`, the provider registry, the capability table) mean new
providers/models are data rows, not code branches. Every client normalizes content to a
plain string and forwards a whitelisted set of kwargs to the underlying SDK; config is
overridable by environment variables with type coercion; `.env` is loaded once at import
with `override=False`.
