from intermediate_representation.sem2sql.infer_from_clause import infer_from_clause
from intermediate_representation.sem2sql.sem2SQL import build_graph

table_names = {'cars_data': 'T1', 'continents': 'T2', 'countries': 'T3'}
schema = {'column_names': [[-1, '*'], [0, 'cont id'], [0, 'continent'], [1, 'country id'], [1, 'country name'], [1, 'continent'], [2, 'id'], [2, 'maker'], [2, 'full name'], [2, 'country'], [3, 'model id'], [3, 'maker'], [3, 'model'], [4, 'make id'], [4, 'model'], [4, 'make'], [5, 'id'], [5, 'mpg'], [5, 'cylinders'], [5, 'edispl'], [5, 'horsepower'], [5, 'weight'], [5, 'accelerate'], [5, 'year']], 'column_names_original': [[-1, '*'], [0, 'ContId'], [0, 'Continent'], [1, 'CountryId'], [1, 'CountryName'], [1, 'Continent'], [2, 'Id'], [2, 'Maker'], [2, 'FullName'], [2, 'Country'], [3, 'ModelId'], [3, 'Maker'], [3, 'Model'], [4, 'MakeId'], [4, 'Model'], [4, 'Make'], [5, 'Id'], [5, 'MPG'], [5, 'Cylinders'], [5, 'Edispl'], [5, 'Horsepower'], [5, 'Weight'], [5, 'Accelerate'], [5, 'Year']], 'column_types': ['text', 'number', 'text', 'number', 'text', 'number', 'number', 'text', 'text', 'text', 'number', 'number', 'text', 'number', 'text', 'text', 'number', 'text', 'number', 'number', 'text', 'number', 'number', 'number'], 'db_id': 'car_1', 'foreign_keys': [[5, 1], [9, 3], [11, 6], [14, 12], [16, 13]], 'primary_keys': [1, 3, 6, 10, 13, 16], 'table_names': ['continents', 'countries', 'car makers', 'model list', 'car names', 'cars data'], 'table_names_original': ['continents', 'countries', 'car_makers', 'model_list', 'car_names', 'cars_data'], 'schema_content_clean': ['*', 'cont id', 'continent', 'country id', 'country name', 'continent', 'id', 'maker', 'full name', 'country', 'model id', 'maker', 'model', 'make id', 'model', 'make', 'id', 'mpg', 'cylinders', 'edispl', 'horsepower', 'weight', 'accelerate', 'year'], 'col_table': [-1, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], 'col_set': ['*', 'cont id', 'continent', 'country id', 'country name', 'id', 'maker', 'full name', 'country', 'model id', 'model', 'make id', 'make', 'mpg', 'cylinders', 'edispl', 'horsepower', 'weight', 'accelerate', 'year'], 'schema_content': ['*', 'ContId', 'Continent', 'CountryId', 'CountryName', 'Continent', 'Id', 'Maker', 'FullName', 'Country', 'ModelId', 'Maker', 'Model', 'MakeId', 'Model', 'Make', 'Id', 'MPG', 'Cylinders', 'Edispl', 'Horsepower', 'Weight', 'Accelerate', 'Year']}
columns = [('none', 'Id', 'cars data'), ('none', 'Continent', 'continents'), ('none', 'CountryName', 'countries')]

graph = build_graph(schema)
print("***graph***", *graph.edges, sep='\n')
from_clause = infer_from_clause(table_names, graph, columns)
print(from_clause)
assert from_clause == "FROM cars_data AS T1 JOIN continents AS T2 JOIN countries AS T3", f"AssertError: {from_clause}"

"""
FROM cars_data AS T1 
JOIN car_names AS T4 ON T1.Id = T4.MakeId 
JOIN model_list AS T5 ON T4.Model = T5.Model 
JOIN car_makers AS T6 ON T5.Maker = T6.Id 
JOIN countries AS T3 ON T6.Country = T3.CountryId 
JOIN continents AS T2 ON T3.Continent = T2.ContId
"""