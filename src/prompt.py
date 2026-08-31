from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RetrievalModel:

    def __init__(self, tokenizer='ai-forever/ruT5-base', model='ai-forever/ruT5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="cpu")

    def transform(self, prompt: str):
        model_prompt = 'rewrite: ' + prompt
        return self.model.generate(model_prompt)
