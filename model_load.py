import math

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import google.generativeai as genai
import os 
import instructor
# Reward model (lower score = more adversarial)
# reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
# reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name).eval().to("cuda")
# reward_tok   = AutoTokenizer.from_pretrained(reward_name)




class ModelLoader: 
    def __init__(self, reward_model_name, gen_model_name, attack_model_name):
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  
        self.gen_model = instructor.from_gemini(
            client=genai.GenerativeModel(gen_model_name),
            mode=instructor.Mode.GEMINI_JSON
        )

        self.attack_model = genai.GenerativeModel(attack_model_name)
