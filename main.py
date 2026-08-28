from core.stor import Stor
from core.llm import Ollama, OllamaRAGModels, format_context
s = Stor(offline=True)
#s.append('test', 'testfile.txt')
result = s.text_query('test','сколько живет человек в городе')
#print(s.list_collections())

o = Ollama(OllamaRAGModels.LIGHT_GEMMA_3_4B, guidelines='отвечай в стиле аниме девочка, используй ^^ чтобы подчеркнуть милоту')

print(o.generate('сколько живет человек в городе', context=format_context(result)))