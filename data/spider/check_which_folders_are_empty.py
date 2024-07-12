import os

destination_path = 'C:/Users/gmess/Desktop/EKPA/2o/Database_systems/2024/Project/ScienceBenchmark/data/spider/generative'

files = os.listdir(destination_path)

files = set(files)
files.discard('generative_schema.json')
files.discard('synthetic_queries.json')
files = list(files)

empty = 0
full = 0

for f in range(len(files)):
    path = destination_path + '/' + files[f]
    
    if os.listdir(path) == []:
        empty = empty + 1
    else:
        full = full + 1

print('A number of ' + str(empty) + ' folders are totally empty.')
print('A number of ' + str(full) + ' folders have files inside.')