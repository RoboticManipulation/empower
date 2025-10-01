from openai import OpenAI
from utils.config import *
import base64
import cv2
import json
import os
import numpy as np

VISUAL_PROMPT_PATH = PROMPT_DIR+"visual_agent_prompt.txt"
DESCRIPTION_PROMPT_PATH = PROMPT_DIR+"description_agent_prompt.txt"
PLANNER_SYSTEM_PROMPT_PATH = PROMPT_DIR+"planner_agent_system_prompt.txt"
PLANNER_USER_PROMPT_PATH = PROMPT_DIR+"planner_agent_user_prompt.txt"

client = OpenAI()
    
def llm_call(system_prompt, user_prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"{system_prompt}"},
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    )

    return (completion.choices[0].message.content)

def vlm_call(prompt, encoded_image):
    agent = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
        "role": "user",
        "content": [
            {
            "type": "text",
                "text":f"{prompt}"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
            },
            },
        ],
        }
    ],
    temperature=0.1,
    )
    response = (agent.choices[0].message.content)
    return response

def image_to_buffer(image):
    if os.path.isfile(image):
        with open(image, "rb") as f:
            encoded_image =  base64.b64encode(f.read()).decode('utf-8')
    else:
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode("utf-8")
    return encoded_image


def multi_agent_planning(image, task_description):

    encoded_image = image_to_buffer(image)

    visual_prompt = open(VISUAL_PROMPT_PATH).read()
    visual_prompt = visual_prompt.replace("<TASK_DESCRIPTION>", task_description)
    vlm_answer = vlm_call(visual_prompt, encoded_image)

    
    print(vlm_answer)


    description_prompt = open(DESCRIPTION_PROMPT_PATH).read()
    description_prompt = description_prompt.replace("<ENVIRONMENT_INFO>", vlm_answer)
    description_prompt = description_prompt.replace("<TASK_DESCRIPTION>", task_description)

    description_agent_answer = vlm_call(description_prompt, encoded_image)
    print(description_agent_answer)

    planner_system_prompt = open(PLANNER_SYSTEM_PROMPT_PATH).read()
    planner_system_prompt = planner_system_prompt.replace("<DESCRIPTION_AGENT>", description_agent_answer)

    planner_user_prompt = open(PLANNER_USER_PROMPT_PATH).read()
    planner_user_prompt = planner_user_prompt.replace("<TASK_DESCRIPTION>", task_description)

    planner_agent_answer = llm_call(planner_system_prompt, planner_user_prompt)

    print(planner_agent_answer)
    return vlm_answer, description_agent_answer,planner_agent_answer 

