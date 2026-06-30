import os
from pathlib import Path
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

from utils.common_utils import get_config, read_yaml

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_api_key(env_var: str) -> str:
    """Read an API key from the environment."""
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise RuntimeError(
            f"{env_var} is not set. Export it before running Empower."
        )
    return api_key


def _canonical_provider(provider: str) -> str:
    normalized = str(provider).strip().lower()
    if normalized in {"chatgpt", "openai", "gpt"}:
        return "chatgpt"
    if normalized == "mistral":
        return "mistral"
    if normalized in {"openrouter", "open_router"}:
        return "openrouter"
    raise ValueError(
        f"Unsupported provider: '{provider}'. Use 'chatgpt', 'mistral', or 'openrouter'."
    )


def _build_llm(provider: str, llm_cfg: dict, vision: bool = False):
    """Instantiate a LangChain chat model for *provider*.

    Args:
        provider:   "chatgpt", "mistral", or "openrouter"
        llm_cfg:    contents of configs/llm/<provider>.yaml
        vision:     True  → use the vision-capable model variant
                    False → use the text-only planning model variant
    """
    model_key = "vision_model" if vision else "model"
    model_name = llm_cfg[model_key]

    if provider == "chatgpt":
        api_key = _resolve_api_key("OPENAI_API_KEY")
        kwargs = dict(
            model=model_name,
            api_key=api_key,
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )
        if "seed" in llm_cfg:
            kwargs["seed"] = llm_cfg["seed"]
        return ChatOpenAI(**kwargs)

    elif provider == "openrouter":
        api_key = _resolve_api_key("OPENROUTER_API_KEY")
        kwargs = dict(
            model=model_name,
            api_key=api_key,
            base_url=llm_cfg.get("base_url", _OPENROUTER_BASE_URL),
        )
        if "max_tokens" in llm_cfg:
            kwargs["max_tokens"] = llm_cfg["max_tokens"]
        if "temperature" in llm_cfg:
            kwargs["temperature"] = llm_cfg["temperature"]
        if "seed" in llm_cfg:
            kwargs["seed"] = llm_cfg["seed"]
        return ChatOpenAI(**kwargs)

    elif provider == "mistral":
        api_key = _resolve_api_key("MISTRAL_API_KEY")
        return ChatMistralAI(
            model=model_name,
            api_key=api_key,
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            "Set llm_provider to 'chatgpt', 'mistral', or 'openrouter' "
            "in configs/llm_config.yaml."
        )


def _image_message(text_prompt: str, encoded_image: str) -> HumanMessage:
    """Build a HumanMessage with an inlined base64 image (works for both providers)."""
    return HumanMessage(content=[
        {"type": "text", "text": text_prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        },
    ])


# ---------------------------------------------------------------------------
# Prompt strings (centralised so single_agent and multi-agent share them)
# ---------------------------------------------------------------------------

_RELATIONS_INSTRUCTIONS = (
    "Use specific relation to describe the position of the objects in the scene. "
    "Do not use 'next to' but you must use 'right to', 'left to', 'behind to', 'beside to', 'on'.\n"
    "For example, if in a scene there is a door, a table in front of the door and a book on the table "
    "with a pen right to it, your answer should be:\n"
    "1) (table, in front of, door)\n"
    "2) (book, on, table)\n"
    "3) (pen, on, table)\n"
    "4) (pen, right to, book)."
)

_ACTION_INSTRUCTIONS = (
    "You must use only the following actions for the plan and nothing else:\n"
    "NAVIGATE : for the movement in the scene towards a point far from you, "
    "for example 'NAVIGATE to the table'\n"
    "GRAB : for the action of picking up an object and specifying which object to grab, "
    "for example 'GRAB bottle'\n"
    "DROP : for the action of placing an object, specifying where with respect to another object, "
    "for example 'DROP bottle left to mug' or 'DROP mug right to bottle' or 'DROP pen into bag'\n"
    "PULL : for the action of pulling an object with the gripper.\n"
    "PUSH : for the action of pushing an object on the ground with the base to free its trajectory "
    "if necessary.\n"
    "Write only the actions for the plan and nothing else."
)

_DROP_ONLY_ACTION_INSTRUCTIONS = (
    "You must use only one action for the plan and nothing else:\n"
    "DROP : place the already held object with respect to a visible reference object. "
    "Use only the placement relations left, right, or on, for example "
    "'DROP bottle left to mug', 'DROP bottle right to mug', or 'DROP bottle on table'.\n"
    "Write only one DROP action line and nothing else."
)


def _action_instructions_for_task(task_description: str) -> str:
    if "Use only one action line: DROP" in task_description:
        return _DROP_ONLY_ACTION_INSTRUCTIONS
    return _ACTION_INSTRUCTIONS


def _is_semantic_placement_task(task_description: str) -> bool:
    return (
        "Use only one action line: DROP" in task_description
        or "place the grasped object where it semantically belongs" in task_description
    )


_ROBOT_CONTEXT = (
    "You are a mobile robot with a base that allows you to move around the environment.\n"
    "You have a robotic arm with a gripper that allows you to pick up and place one object at a time.\n"
    "Work as a Markovian agent, so you can only see the last action and the current state of the "
    "environment. After each step, update the state of the environment to elaborate the next step "
    "executable in the updated environment."
)


# ---------------------------------------------------------------------------
# Agents class
# ---------------------------------------------------------------------------

class Agents:
    """LangChain-based robot task-planning agents.

    Supports ChatGPT (GPT-4o), Mistral (via Mistral AI API / Pixtral for vision),
    and OpenRouter's OpenAI-compatible API.
    The active provider and model parameters are read from:
        configs/llm_config.yaml      — provider selection
        configs/llm/<provider>.yaml  — model-specific parameters
    API keys are read from OPENAI_API_KEY, MISTRAL_API_KEY, or OPENROUTER_API_KEY
    environment variables.

    Args:
        image:            Base64-encoded JPEG from the robot's camera.
        task_description: Natural-language description of the task to solve.
    """

    def __init__(
        self,
        image: str,
        task_description: str,
        llm_provider: str | None = None,
        *,
        llm_cfg: dict | None = None,
        environment_task_description: str | None = None,
    ):
        self.encoded_image = image
        self.task_description = task_description
        self.environment_task_description = (
            environment_task_description
            if environment_task_description is not None
            else task_description
        )

        master_cfg = get_config("llm_config")
        self.provider = _canonical_provider(llm_provider or master_cfg["llm_provider"])

        if llm_cfg is None:
            llm_cfg_path = _ROOT / "configs" / "llm" / f"{self.provider}.yaml"
            if not llm_cfg_path.exists():
                raise FileNotFoundError(
                    f"LLM config not found: {llm_cfg_path}\n"
                    f"Expected a file named '{self.provider}.yaml' in configs/llm/."
                )
            llm_cfg = read_yaml(llm_cfg_path)
        else:
            llm_cfg = dict(llm_cfg)
            llm_cfg.setdefault("vision_model", llm_cfg.get("model"))

        # Two LLM instances: one with vision for scene understanding, one text-only for planning.
        # For ChatGPT the same model handles both; for Mistral, Pixtral handles vision.
        self._vision_llm = _build_llm(self.provider, llm_cfg, vision=True)
        self._text_llm = _build_llm(self.provider, llm_cfg, vision=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_vision(self, prompt: str) -> str:
        return self._vision_llm.invoke([
            _image_message(prompt, self.encoded_image)
        ]).content

    def _invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        return self._text_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]).content

    # ------------------------------------------------------------------
    # Public agent methods
    # ------------------------------------------------------------------

    def single_agent(self) -> str:
        """Single LLM call: scene understanding + action planning from one image.

        Returns:
            String with sections '***RELATIONS***' and '***PLAN***'.
        """
        action_instructions = _action_instructions_for_task(self.task_description)
        prompt = (
            f"{_ROBOT_CONTEXT}\n\n"
            "You are also very capable of describing a scene provided an image as input.\n"
            "From the image, produce a set of relations in the form of a triple "
            "(subject, relation, object).\n"
            f"Write just the triples that are essential to solve the following task: {self.task_description}\n"
            f"{_RELATIONS_INSTRUCTIONS}\n\n"
            "For the same task given in input, plan a sequence of actions to solve the task.\n"
            "Use univocal names given in the relations of the environment to specify objects.\n\n"
            f"{action_instructions}\n\n"
            "The output must follow this format exactly:\n"
            "***RELATIONS***\n"
            "<list of relation triples>\n"
            "***PLAN***\n"
            "<list of action steps>"
        )
        return self._invoke_vision(prompt)

    def multi_agent_vision_planning(self) -> Tuple[str, str, str]:
        """Three-stage pipeline: environment agent → description agent → planning agent.

        Stage 1 (vision LLM): Extract spatial relation triples from the image.
        Stage 2 (vision LLM): Build a high-level scene description using Stage 1 names.
        Stage 3 (text LLM):   Generate the action plan using the Stage 2 description.
                              This stage can use Mistral (text-only) when configured.

        Returns:
            Tuple of (environment_info, description_info, plan).
        """
        # --- Stage 1: environment agent ---
        env_prompt = (
            "You are an assistant able to accurately describe the content of an image.\n"
            "Capture the main objects present and provide all spatial relations between them.\n"
            "Answer only with triples in the form (subject, relation, object) — nothing else.\n"
            f"Write just the triples essential to solve the following task: "
            f"{self.environment_task_description}\n"
            "IMPORTANT: Use full, descriptive object names that include the object type "
            "(e.g. 'coca-cola bottle', 'monster energy drink can', 'beer bottle', "
            "'paper towel roll', 'spray bottle', 'windex spray bottle'). "
            "Do NOT use brand names alone or hyphenated abbreviations. "
            "Each object name must be recognisable as a visual category.\n"
            f"{_RELATIONS_INSTRUCTIONS}"
        )
        if _is_semantic_placement_task(self.task_description):
            env_prompt += (
                "\nFor semantic placement, list every visible movable object on the "
                "placement surface using relation triples only. Never output action "
                "lines such as DROP, GRAB, NAVIGATE, or GRAB/PLACE commands. "
                "Do not name objects that are not visible in the image."
            )
        environment_info = self._invoke_vision(env_prompt)

        # --- Stage 2: description agent ---
        desc_prompt = (
            "You are an assistant able to accurately describe the content of an image.\n"
            "Describe the image so that someone can fully understand the scene without seeing it.\n"
            f"Use only the object names from these relations: {environment_info} "
            "— do not add adjectives.\n"
            "Give a high-level description and precise instructions to solve "
            f"the following task: {self.task_description}. "
            "Minimise the number of steps and find the best plan.\n"
            "If the task is ambiguous (e.g. multiple objects of the same type), "
            "specify the object's position relative to other objects."
        )
        description_info = self._invoke_vision(desc_prompt)

        # --- Stage 3: planning agent (text only — benefits from Mistral's reasoning) ---
        system_prompt = (
            f"{_ROBOT_CONTEXT}\n\n"
            "You have the following detailed scene description and preliminary instructions "
            f"to help you define the plan:\n{description_info}\n"
            "Use this information as a guide only."
        )
        user_prompt = (
            f"The task is: {self.task_description}\n\n"
            f"{_action_instructions_for_task(self.task_description)}"
        )
        if _is_semantic_placement_task(self.task_description):
            user_prompt += (
                "\nThe DROP reference object must be copied exactly from one object "
                "name in these environment relations. Do not invent objects that are "
                "not listed there:\n"
                f"{environment_info}"
            )
        plan = self._invoke_text(system_prompt, user_prompt)

        return environment_info, description_info, plan
