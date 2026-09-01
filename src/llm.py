from dataclasses import dataclass
import ollama
from src.ollamas import OllamaRAGModels
from chromadb.api.types import QueryResult
def format_prompt(template: str, content: list[str], user_prompt: str):
    return template.replace('{content}','\n\n\n'.join(content)).replace('{prompt}', user_prompt)

def format_context(content: QueryResult) -> list[str]:
    return '\n'.join(content.get('documents')[0])

def to_text(model_response: ollama.ChatResponse):
    return f"{model_response.model}: `{model_response.message.content}`"

@dataclass
class OllamaRAG:
    model_name: OllamaRAGModels | str
    guidelines: str = 'Отвечай строго по документу. Без контекста - "Не знаю". Никаких домыслов.'
    template: str = "Документ: {content}\nВопрос: {prompt}\nОтвет (только по контексту). Обязательно приводи цитаты из документа для обоснования своей позиции.:"


    def generate(self, prompt: str, context: str):
        system = dict(role='system', content=self.guidelines)
        if not context:
            final = dict(role='user', content=prompt)
        else:
            final = dict(role='user', content= self._format_prompt(context, prompt))
        return ollama.chat(model=self.model_name, messages=[system, final])

    def _format_prompt(self, content: str, user_prompt: str):
        return self.template.replace('{content}',content).replace('{prompt}', user_prompt)
