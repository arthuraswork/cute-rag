from dataclasses import dataclass
import ollama
from src.ollamas import OllamaRAGModels
from chromadb.api.types import QueryResult
def format_prompt(template: str, content: list[str], user_prompt: str):
    return template.replace('{content}','\n\n\n'.join(content)).replace('{prompt}', user_prompt)

def format_context(content: QueryResult) -> list[str]:
    documents, ids = content.get('documents'), content.get('ids')
    return [f"FROM {doc_id} BEGIN: {document} END;" for document ,doc_id in zip(documents, ids)]

@dataclass
class Ollama:
    model_name: OllamaRAGModels | str
    guidelines: str 
    template: str = "Используй `{content}`; чтобы дать правильный ответ на запрос пользователя `{prompt}`;, если нет точного ответа, отвечай - я не знаю"


    def generate(self, prompt: str, context: list[str]):
        system = dict(role='system', content=self.guidelines)
        if not context:
            final = dict(role='user', content=prompt)
        else:
            final = dict(role='user', content= self._format_prompt(context, prompt))
        return ollama.chat(model=self.model_name, messages=[system, final])

    def _format_prompt(self, content: list[str], user_prompt: str):
        return self.template.replace('{content}','\n\n\n'.join(content)).replace('{prompt}', user_prompt)
