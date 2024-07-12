import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import openai
from typing import List

from common_query_types import common_query_types
from group_pairs_to_find_templates import group_query_types, map_semql_actions_only
from sample_queries.sample_query import sample_query
from tools.transform_generative_schema import GenerativeSchema
from dotenv import load_dotenv

load_dotenv()

"""
Synthetic data generator by using Ursin's templates
"""

def ask_gpt(sample: str, number_of_choices: int, model_id: str, sec=3):

    # PROMPT ENGINEERING & RESPONSE PARAMETERS OPTIMIZATION
    response = None
    prompt = sample
    while response is None:
        try:
            response = openai.Completion.create(
                model=model_id,
                prompt=prompt + '. For the given SQL query, write only the semantically equivalent natural language query in a single sentence.',
                top_p=0.9,             # Makes model less selective (increases diversity)
                max_tokens=100,
                n=number_of_choices,
                frequency_penalty=0.8, # Makes model more diverse responses
                presence_penalty=0.5,  # We achieve variations in phrasing & maintaining coherence
                #stop=["\n"]
            )
            time.sleep(sec)
            print(f"sleep {sec} sec...")
        except Exception as e:
            print(f"{e}")
            sec += sec
            time.sleep(sec)
            pass
    print(response)
    return response, prompt


def main(args):

    with open(Path(args.data_path) / 'original' / 'tables.json') as f:
        schemas = json.load(f)
        original_schema = schemas[0]  # We assume there is only one db-schema in this file

    generative_schema = GenerativeSchema(Path(args.data_path) / 'generative' / 'generative_schema.json')

    db_config = SimpleNamespace(database=args.database,
                                db_user=args.db_user,
                                db_password=args.db_password,
                                db_host=args.db_host,
                                db_port=args.db_port,
                                db_options=args.db_options,
                                path=None) # Path attribute added by us

    query_cache = []

    for idx, (query_type, multiplier) in enumerate(common_query_types().items()):

        round_idx = 0
        fail_to_sample = 0

        # We might have to repeat the sampling process multiple times to get enough samples (exceptions due to unfavorable samplings),
        # but we still don't want to be caught in an infinite loop
        while round_idx < (args.base_number_of_samples_per_query_type * multiplier) and fail_to_sample < 50:

            try:
                sampled_query, sampled_query_replaced = sample_query(query_type, original_schema, generative_schema, db_config)

                if sampled_query in query_cache:
                    raise ValueError('Query already sampled')
                else:
                    query_cache.append(sampled_query)

                print(f'{query_type}                        {sampled_query}')

                response, prompt = ask_gpt(sampled_query_replaced, args.number_of_choices, args.gpt_model)

                gpt_choices = [f"({idx}) {c['text'].strip()}" for idx, c in enumerate(response['choices'])]

                with open(Path(args.output_folder) / f'{idx}_{round_idx}.txt', 'w', encoding='utf-8') as f:
                    f.write(prompt + '\n\n###\n\n')
                    f.write('\nOriginal Query:\n')
                    f.write(sampled_query)
                    f.write('\nGPT-3 choices:\n')
                    f.write('\n'.join(gpt_choices))

                round_idx += 1

            except ValueError as e:
                print(f'Exception:{e}')
                fail_to_sample += 1

        print()
        print()


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--data_path', type=str, default='data/cordis')
    arg_parser.add_argument('--output_folder', type=str, default='data/cordis/generative/generated')
    arg_parser.add_argument('--database', type=str, default='cordis_temporary')
    arg_parser.add_argument('--db_options', type=str, default=f"-c search_path=unics_cordis,public")

    # CORDIS --db_options: default=f"-c search_path=unics_cordis,public"
    # ONCOMX --db_options: default=f"-c search_path=oncomx_v1_0_25,public"
    # SDSS --db_options: default=f"-c search_path=lite,public"

    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--base_number_of_samples_per_query_type', type=int, default=50, help='The base number of samples per query type. This number, multiplied with the query type multiplier (see "common_query_types.py") is the total number of samples that will be generated for each query type.')
    arg_parser.add_argument('--db_user', type=str, default='postgres')
    arg_parser.add_argument('--db_password', type=str, default='geme1997')
    arg_parser.add_argument('--db_host', type=str, default='localhost')
    arg_parser.add_argument('--db_port', type=str, default='5433')
    arg_parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo-instruct')

    args = arg_parser.parse_args()
    main(args)