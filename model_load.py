import math

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import os 
import instructor
from openai import OpenAI
# Reward model (lower score = more adversarial)
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
# reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name).eval().to("cuda")
# reward_tok   = AutoTokenizer.from_pretrained(reward_name)




class ModelLoader: 
    def __init__(self, reward_model_name):
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

        self.gen_model = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  
    ),
    mode=instructor.Mode.JSON,
)
        self.attack_model = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  
            ),
            mode=instructor.Mode.JSON,
        )
