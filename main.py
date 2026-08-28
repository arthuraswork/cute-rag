from core.stor import Stor
s = Stor(offline=True)
#s.append('test', 'testfile.txt')
print(s.text_query('test','сколько живет человек в городе'))
print(s.list_collections())