import argparse
import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import language_tool_python
from textstat import textstat

tool = language_tool_python.LanguageTool('en-US')

def read_generative_choices(path):
    with open(path, encoding='utf-8') as f:
        lines = f.readlines()
    try:
        original_query = lines[lines.index("Original Query:\n") + 1]
    except:
        original_query = '---'

    generative_choices = []
    for i in range(0, 8):
        choice_line = filter(lambda l: l.startswith(f'({i})'), lines)
        choice = next(choice_line)[4:]
        choice = choice.strip()
        # 1. Remove the empty lines if existed
        # 2. Always capitalize the first letter of a sentences so that we can eliminate some deduplicated results
        if len(choice) > 0:
            _choice = choice[0].upper() + choice[1:]
            if _choice not in generative_choices:
                generative_choices.append(_choice)

    return generative_choices, original_query.strip(), lines

def original_names_to_names():
    # Here we create a dictionary mapping between original names and names
    with open(args.generative_schema_path, 'r', encoding='utf-8') as gsf:
        schema_data = json.load(gsf)

        number_of_columns_in_schema = 0
        for i in range(len(schema_data)):
            number_of_columns_in_schema += len(schema_data[i]["columns"])

        table_names_mapping, column_names_mapping = dict(), dict()
        for i in range(len(schema_data)):
            table_names_mapping[str(schema_data[i]["original_name"])] = str(schema_data[i]["name"])
            for j in range(len(schema_data[i]["columns"])):
                column_names_mapping[str(schema_data[i]["columns"][j]["original_name"])] = str(schema_data[i]["columns"][j]["name"])

    return table_names_mapping, column_names_mapping

def is_not_number_string(s):
    try:
        int(s)  # Try to convert the string to an int
        return False  # If successful, it's a number, so return False
    except ValueError:
        return True  # If conversion fails, it's not a number, so return True

def processing_sql_query(sql_query, table_names_mapping, column_names_mapping):
    # SQL PROCESSING
    # Split and remove sql keywords
    sql_query = sql_query.lower()
    chars_to_exclude = ["\n", "\'", "\"", " ", "."]
    sql_keywords = [
        "add", "all", "alter", "and", "any", "as", "asc", "backup", "between", "case",
        "check", "column", "constraint", "create", "database", "default", "delete",
        "desc", "distinct", "drop", "exec", "exists", "foreign", "from", "full", "group",
        "having", "in", "index", "inner", "insert", "into", "is", "join", "key", "left",
        "like", "limit", "not", "null", "or", "order", "outer", "primary", "procedure",
        "right", "rownum", "select", "set", "table", "top", "truncate", "union",
        "unique", "update", "values", "view", "where"
    ]

    sql_tokens = re.split(r'[ .,=<>\'\"]+', sql_query)

    sql_tokens = list(filter(is_not_number_string, sql_tokens))
    clean_sql_tokens = [item for item in sql_tokens if item not in sql_keywords]
    # Keep unique original names in list
    unique_sql_tokens = list(set(clean_sql_tokens))

    # Replace original names (tokens) to names, if exist
    final_sql_schema_name_tokens = []
    final_sql_other_tokens = []
    for item in unique_sql_tokens:
        if item in table_names_mapping:
            final_sql_schema_name_tokens.append(table_names_mapping[item])
        elif item in column_names_mapping:
            final_sql_schema_name_tokens.append(column_names_mapping[item])
        else:
            final_sql_other_tokens.append(item)

    final_sql_schema_name_tokens = [item.split(" ") for item in final_sql_schema_name_tokens]
    final_sql_schema_name_tokens = sum(([item] if not isinstance(item, list) else item for item in final_sql_schema_name_tokens), [])
    final_sql_schema_name_tokens = list(filter(lambda x: x not in chars_to_exclude, final_sql_schema_name_tokens))
    # Keep unique sql schema names tokens
    final_sql_schema_name_tokens = list(set(final_sql_schema_name_tokens)) 

    final_sql_other_tokens = list(filter(lambda x: x not in chars_to_exclude, final_sql_other_tokens))
    # Keep unique sql other tokens
    final_sql_other_tokens = list(set(final_sql_other_tokens))  

    return final_sql_schema_name_tokens, final_sql_other_tokens

def processing_nlq(nlq):
    # NLQ PROCESSING
    chars_to_exclude = ["\n", "\'", "\"", " ", "."]

    nlq_lower = nlq.lower()
    nlq_tokens = re.split(r'[ \'\"]+', nlq_lower)
    nlq_tokens = list(filter(lambda x: x not in chars_to_exclude, nlq_tokens))
    # Keep unique nlq tokens
    final_nlq_tokens = list(set(nlq_tokens))

    return final_nlq_tokens
def schema_related_score(sql_query, nlq, table_names_mapping, column_names_mapping):
    # Here we search if NLQ tokens exist in SQL tokens
    final_sql_schema_names_tokens, final_sql_other_tokens = processing_sql_query(sql_query, table_names_mapping, column_names_mapping)
    final_nlq_tokens = processing_nlq(nlq)

    schema_score = 0
    for item in final_nlq_tokens:
        if item in final_sql_schema_names_tokens:
            schema_score += 2
        elif item in final_sql_other_tokens:
            schema_score += 1

    return schema_score

def grammar_and_readability_score(nlq):
    # Grammar score
    matches = tool.check(nlq)
    grammar_score = len(matches)

    # Readability score
    readability_score = textstat.flesch_kincaid_grade(nlq)

    # Combined score (lower is better)
    # But we transformed the score to reverse the lower
    # to higher value in order two scores to be consistent
    return 1/(grammar_score + readability_score)

def rank_choices(sql_query, choices: List[str], table_names_mapping, column_names_mapping):
    choice_scores = {s: 0 for s in choices}
    for nlq in choices:
        grammar_and_readability_final_score = grammar_and_readability_score(nlq)
        schema_related_final_score = schema_related_score(sql_query, nlq, table_names_mapping, column_names_mapping)

        choice_scores[nlq] = 0.3 * grammar_and_readability_final_score + 0.7 * schema_related_final_score

    return choice_scores

def print_reranked(file_path: Path, original_lines: List[str], choices_reranked: List[Tuple[str, float]]):
    re_ranked = [f'{v:.3f}  {c}\n' for c, v in choices_reranked]
    # We add ranked files separately from generated files (please create 'ranked' folder into 'generative' folder)
    new_file = Path(file_path.parent.parent / 'ranked' / f'_{file_path.name}')

    new_file_content = f"""{''.join(original_lines)}


Re-ranked choices:
{''.join(re_ranked)}
"""

    new_file.write_text(new_file_content, encoding='utf-8')

def retrieve_index_and_numberid_from_txt(txt_filepath):
    txt_name = txt_filepath.name[:-4]
    matches = re.findall(r'(\d)_(\d+)', txt_name)
    idx = matches[0][0]
    number_id = matches[0][1]

    return idx, number_id

def main(args):
    samples = []
    table_names_mapping, column_names_mapping = original_names_to_names()
    scores_storing = []
    for idx, path in enumerate(Path(args.input_data_path).glob('*.txt')):
        choices, original_sql_query, original_file_content = read_generative_choices(path)
        # Here we use our method for evaluating and reranking choices
        choice_scores = rank_choices(original_file_content[0], choices, table_names_mapping, column_names_mapping)
        choice_reranked = {k: v for k, v in sorted(choice_scores.items(), key=lambda item: item[1], reverse=True)}
        choice_reranked_list = list(choice_reranked.items())

        index, number_id = retrieve_index_and_numberid_from_txt(path)
        highest_score_value = next(iter(choice_reranked.values()))
        scores_storing.append((index, number_id, highest_score_value))

        print_reranked(path, original_file_content, choice_reranked_list)

        print(f'{idx}: {original_sql_query}')
        print(choices)
        print(choice_reranked)
        print()
        print()

        # We wanna keep both, the first and the second choice after re-ranking
        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_1',
            'user': "gpt-3",
            'question': choice_reranked_list[0][0],
            'query': original_sql_query
        })

        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_2',
            'user': "gpt-3",
            'question': choice_reranked_list[1][0],
            'query': original_sql_query
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    
    # Save results to an excel file
    df_scores = pd.DataFrame(scores_storing, columns=['Index', 'Number', 'Score'])
    df_scores = df_scores.sort_values(by=['Index', 'Number'])
    df_scores.to_excel(args.output_scores, index=False)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--input_data_path', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--generative_schema_path', type=str, default='data/cordis/generative/generative_schema.json')
    arg_parser.add_argument('--output_file', type=str, default='data/cordis/generative/all_synthetic.json')
    arg_parser.add_argument('--output_scores', type=str, default='data/cordis/generative/output_scores_cordis.xlsx')
    arg_parser.add_argument('--db_id', type=str, default='cordis_temporary')

    args = arg_parser.parse_args()
    main(args)