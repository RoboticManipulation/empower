import os
import yaml
from pathlib import Path
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_MASTER_CFG_PATH = _ROOT / "configs" / "llm_config.yaml"


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_api_key(yaml_value, env_var: str):
    """Use YAML key if set to a non-placeholder value, else ``os.environ[env_var]``."""
    if yaml_value is None:
        return os.environ.get(env_var)
    s = str(yaml_value).strip()
    if not s or s == "...":
        return os.environ.get(env_var)
    return s


def _build_llm(provider: str, master_cfg: dict, llm_cfg: dict, vision: bool = False):
    """Instantiate a LangChain chat model for *provider*.

    Args:
        provider:   "openai" or "mixtral"
        master_cfg: contents of configs/llm_config.yaml
        llm_cfg:    contents of configs/llm/<provider>.yaml
        vision:     True  → use the vision-capable model variant
                    False → use the text-only planning model variant
    """
    model_key = "vision_model" if vision else "model"
    model_name = llm_cfg[model_key]

    if provider == "openai":
        api_key = _resolve_api_key(master_cfg.get("openai_api_key"), "OPENAI_API_KEY")
        kwargs = dict(
            model=model_name,
            api_key=api_key,
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )
        if "seed" in llm_cfg:
            kwargs["model_kwargs"] = {"seed": llm_cfg["seed"]}
        return ChatOpenAI(**kwargs)

    elif provider == "mixtral":
        api_key = _resolve_api_key(master_cfg.get("mistral_api_key"), "MISTRAL_API_KEY")
        return ChatMistralAI(
            model=model_name,
            api_key=api_key,
            max_tokens=llm_cfg["max_tokens"],
            temperature=llm_cfg["temperature"],
        )

    else:
        raise ValueError(
            f"Unsupported provider: '{provider}'. "
            "Set llm_provider to 'openai' or 'mixtral' in configs/llm_config.yaml."
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

    Supports OpenAI (GPT-4o) and Mixtral (via Mistral AI API / Pixtral for vision).
    The active provider and model parameters are read from:
        configs/llm_config.yaml      — provider selection & API keys
        configs/llm/<provider>.yaml  — model-specific parameters

    Args:
        image:            Base64-encoded JPEG from the robot's camera.
        task_description: Natural-language description of the task to solve.
    """

    def __init__(self, image: str, task_description: str):
        self.encoded_image = image
        self.task_description = task_description

        master_cfg = _load_yaml(_MASTER_CFG_PATH)
        self.provider = master_cfg["llm_provider"]

        llm_cfg_path = _ROOT / "configs" / "llm" / f"{self.provider}.yaml"
        if not llm_cfg_path.exists():
            raise FileNotFoundError(
                f"LLM config not found: {llm_cfg_path}\n"
                f"Expected a file named '{self.provider}.yaml' in configs/llm/."
            )
        llm_cfg = _load_yaml(llm_cfg_path)

        # Two LLM instances: one with vision for scene understanding, one text-only for planning.
        # For OpenAI the same model handles both; for Mixtral, Pixtral handles vision.
        self._vision_llm = _build_llm(self.provider, master_cfg, llm_cfg, vision=True)
        self._text_llm = _build_llm(self.provider, master_cfg, llm_cfg, vision=False)

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
        prompt = (
            f"{_ROBOT_CONTEXT}\n\n"
            "You are also very capable of describing a scene provided an image as input.\n"
            "From the image, produce a set of relations in the form of a triple "
            "(subject, relation, object).\n"
            f"Write just the triples that are essential to solve the following task: {self.task_description}\n"
            f"{_RELATIONS_INSTRUCTIONS}\n\n"
            "For the same task given in input, plan a sequence of actions to solve the task.\n"
            "Use univocal names given in the relations of the environment to specify objects.\n\n"
            f"{_ACTION_INSTRUCTIONS}\n\n"
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
                              This stage can use Mixtral (text-only) when configured.

        Returns:
            Tuple of (environment_info, description_info, plan).
        """
        # --- Stage 1: environment agent ---
        env_prompt = (
            "You are an assistant able to accurately describe the content of an image.\n"
            "Capture the main objects present and provide all spatial relations between them.\n"
            "Answer only with triples in the form (subject, relation, object) — nothing else.\n"
            f"Write just the triples essential to solve the following task: {self.task_description}\n"
            "IMPORTANT: Use full, descriptive object names that include the object type "
            "(e.g. 'coca-cola bottle', 'monster energy drink can', 'beer bottle', "
            "'paper towel roll', 'spray bottle', 'windex spray bottle'). "
            "Do NOT use brand names alone or hyphenated abbreviations. "
            "Each object name must be recognisable as a visual category.\n"
            f"{_RELATIONS_INSTRUCTIONS}"
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

        # --- Stage 3: planning agent (text only — benefits from Mixtral's reasoning) ---
        system_prompt = (
            f"{_ROBOT_CONTEXT}\n\n"
            "You have the following detailed scene description and preliminary instructions "
            f"to help you define the plan:\n{description_info}\n"
            "Use this information as a guide only."
        )
        user_prompt = (
            f"The task is: {self.task_description}\n\n"
            f"{_ACTION_INSTRUCTIONS}"
        )
        plan = self._invoke_text(system_prompt, user_prompt)

        return environment_info, description_info, plan
