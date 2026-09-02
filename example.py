from src.stor import Stor
from src.llm import OllamaRAG, OllamaRAGModels, format_context, to_text
import readline

s = Stor(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
k = 8
o = OllamaRAG(OllamaRAGModels.LIGHT_GEMMA_3_4B)
while True:
    try:
        uinput = input('$ ')
        verb = uinput.split()[0]
        match verb:
            case 'help':
                print("""
help - список команд
add  - добавить файл в бд
ask  - спросить ллм на основе документов
ls   - информация о коллекциях
""")
            case 'add':
                _, collection, path = uinput.split(maxsplit=2)
                try:
                    s.append(collection, path)
                    print(path, 'добавлен')
                except Exception as e:
                    print(e)
            case 'ask':
                _, collection, prompt = uinput.split(maxsplit=2)
                context = format_context(s.text_query(collection, prompt, k=k))
                response = to_text(o.generate(prompt, context))
                print(response)
            case 'ls':
                print('Коллекции:')
                print('\n'.join(s.list_collections()))
    except Exception as e:
        print('Ошибка в аргументах:', e)