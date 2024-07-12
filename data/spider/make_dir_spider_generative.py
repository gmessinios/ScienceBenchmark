import os

basic_path = 'C:/Users/gmess/Desktop/EKPA/2o/Database_systems/2024/Project/ScienceBenchmark/data/spider/original/database'
destination_path = 'C:/Users/gmess/Desktop/EKPA/2o/Database_systems/2024/Project/ScienceBenchmark/data/spider/generative'

files = os.listdir(basic_path)

for f in range(len(files)):
    path = destination_path + '/' + files[f]
    os.mkdir(path)