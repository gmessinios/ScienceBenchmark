# I am not the creator of this project, but I make some changes in the third and fourth phase of this project using GPT-3.5-turbo-instruct model with appropriate prompt engineering and another metric function in the discriminative phase.

## NL QL data augmentation

This repo served all codes related to data augmentation approach for NL-to-SQL tasks.

### Components introduction

#### Data folder `/data`

Each subfolder under `data` holds the data for different datasets respectively. After entering the subfolder, the subsubfolders contain the data that stands for,

- `data_aug` for data augmentation files for scripts in `src/data_augmentation`
- `generative` for data augmentation files for scripts in `src/syntehtic_data`
- `handmade_training_data` for generated data to train / evaluate ValueNet
- `original` for containing data schemata for the database

#### Script folder `/src`

Folder `src` holds all scripts for the data augmentation. The following subfolders serve for different puposes,

- `data_augmentation` for data augmentation pipe with seeding data and training CRITIC model.
  
- `synthetic_data` for data augmentation pipe with data generative schema
  
- `intermediate_representation`, `preprocessing` and `spider` for all AST related helper files

- `tools` for all other helper files

### Usage

#### Install dependencies and prerequisites

1. Run `pip install requests`

2. Run `pip install -r requirements.txt` to install all required dependencies

3. Set `src` folder and its parent folder as Sources root

4. Go to [https://github.com/ckosten/sciencebenchmark_dataset/tree/master], and download the .sql.gz files for CORDIS, ONCOMX and SDSS databases.

5. Run the `restore.sql` file for each of the three databases, in order to create them.

6. Sign up in [openai.com](https://openai.com/api/) and configure an API key in the page [https://beta.openai.com/account/api-keys](https://beta.openai.com/account/api-keys)

7. Add the api key as the line shown below in `.env` file under the root folder of the project. If you haven't this file, please create it.

```shell
OPENAI_API_KEY=<api_key_from_openai>
```

#### Data augmentation based on shuffling on AST


##### With schema based and grammar & readability based re-ranking

1. Prepare generative schema
   - Run te script shown as below

   ```bash
   python3 src/tools/transform_generative_schema.py --db_path <datapath>
   ```

   , where `<datapath>` is the path in `data` which contains the subfolder `data/<datapath>/generative` and original schema `data/<datapath>/original/tables.json`

2. Add more query types into `src/synthetic_data/common_query_types` if neccessary
   -- find out query types with

   ```bash
   python3 src/synthetic_data/group_pairs_to_find_templates.py \
   --data_containing_semql <training_or_dev_json_file>
   ```

   -- Add query types found into file `src/synthetic_data/common_query_types.py`

3. Generate synthetical data
   - Write your own generating file running `src/synthetic_data/generate_synthetical_data.py` file
   We generalize this step for the three domain specific ScienceBenchmark databases ('CORDIS', 'ONCOMX', 'SDSS')

   ```bash
   python3 src/synthetic_data/generate_synthetical_data.py \
   --data_path <data_path> \
   --output_folder <output_folder_path> \
   --database <database> \
   --db_options <db_options> \
   --db_user <user> \
   --db_password <password> \
   --db_host <host> \
   --db_port <port> \
   --gpt_model <model>
   ```

4. re-ranking data and generate handmade training data

   ```bash
   python3 src/synthetic_data/apply_sentence_embedding_reranking_new.py \
   --input_data_path <input_data_path> \
   --generative_schema_path <generative_schema_path> \
   --output_file <output_filepath> \
   --output_scores <scores_filepath> \
   --db_id <db_id>
   ```