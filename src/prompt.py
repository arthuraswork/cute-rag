from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class RetrievalModel:

    def __init__(self, tokenizer='ai-forever/ruT5-base', model='ai-forever/ruT5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="cpu")

    def rewrite(self, prompt: str):
        prompt = 'rewrite: ' + prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
