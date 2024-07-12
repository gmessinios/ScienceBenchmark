import argparse
import json
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer, util
from collections import Counter


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
        # 1. remove the empty lines if existed
        # 2. Always capitalize the first letter of a sentences so that we can eliminate some deduplicated results.
        if len(choice) > 0:
            _choice = choice[0].upper() + choice[1:]
            if _choice not in generative_choices:
                generative_choices.append(_choice)

    return generative_choices, original_query.strip(), lines


def rank_by_aggregated_pairwise_similarity(choices: List[str], model: SentenceTransformer):
    paraphrases = util.paraphrase_mining(model, choices)

    choice_scores = {s: 0 for s in choices}
    for paraphrase in paraphrases:
        score, i, j = paraphrase
        choice_scores[choices[i]] += score
        choice_scores[choices[j]] += score

    c = Counter(choice_scores)
    return c.most_common()


def print_reranked(file_path: Path, original_lines: List[str], choices_reranked: List[Tuple[str, float]]):
    re_ranked = [f'{v:.3f}  {c}\n' for c, v in choices_reranked]

    new_file = Path(file_path.parent / f'_{file_path.name}')

    new_file_content = f"""{''.join(original_lines)}


Re-ranked choices:
{''.join(re_ranked)}
"""

    new_file.write_text(new_file_content, encoding='utf-8')

import re
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_text(query, stop_words):
    query = query.lower() # lower text
    query = re.sub(r'[^\w\s]', '', query) # subtract punctuation marks
    words = query.split() # words separation
    words = [word for word in words if word not in stop_words]
    return set(words)

def jaccard_similarity(set_1, set_2):
    sets_intersection = set_1.intersection(set_2)
    sets_union = set_1.union(set_2)
    return len(sets_intersection)/len(sets_union)

def main(args):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    samples = []
    stop_words = {'select', 'from', 'where', 'as'}

    for idx, path in enumerate(Path(args.input_data_path).glob('*.txt')):
        choices, original_sql_query, original_file_content = read_generative_choices(path)

        original_sql_query_words_set = preprocess_text(original_file_content[0], stop_words)
        nlq_words_set = [preprocess_text(nlq, stop_words) for nlq in choices]

        similarity_scores = [(nlq, jaccard_similarity(original_sql_query_words_set, nlq_set)) for nlq, nlq_set in zip(choices, nlq_words_set)]

        choice_reranked = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        #choice_reranked = rank_by_aggregated_pairwise_similarity(choices, model)

        print_reranked(path, original_file_content, choice_reranked)

        print(f'{idx}: {original_sql_query}')
        print(choices)
        print(choice_reranked)
        print()
        print()

        # we wanna keep both, the first and the second choice after re-ranking
        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_1',
            'user': "gpt-3",
            'question': choice_reranked[0][0],
            'query': original_sql_query
        })

        samples.append({
            'db_id': args.db_id,
            'id': f'{idx}_2',
            'user': "gpt-3",
            'question': choice_reranked[1][0],
            'query': original_sql_query
        })

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_data_path', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--output_file', type=str, default='data/cordis/generative/all_synthetic.json')
    arg_parser.add_argument('--db_id', type=str, default='cordis_temporary')

    args = arg_parser.parse_args()
    main(args)