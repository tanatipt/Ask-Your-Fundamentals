import json
import pandas as pd


if __name__ == "__main__":
    # Random seed for reproducibility
    random_state = 12345
    # Configuration for selecting questions
    symbol = ['GS']
    available_years = ['2016','2013', '2012', '2014', '2017']
    output_path = 'data/evaluation_qa/qa_dataset_v1.csv'
    total_q = 60
    # Proportions of different question types
    q_prop = {"answerable" : 0.7, "unanswerable" : 0.15, "negative" : 0.15}

    # Paths to the metadata files containing question data
    file_paths = [
        't2-ragbench/data/FinQA/test/metadata.jsonl',
        't2-ragbench/data/FinQA/dev/metadata.jsonl',
        't2-ragbench/data/FinQA/train/metadata.jsonl',
        't2-ragbench/data/ConvFinQA/turn_0.jsonl',
        't2-ragbench/data/VQAonBD/metadata.jsonl'
    ]

    # Lists to hold dataframes for positive and negative samples
    positive_dfs = []
    negative_dfs = []

    # Process each metadata file and separate questions into positive and negative samples
    for file_path  in file_paths:
        with open(file_path) as f:
            data = [json.loads(line) for line in f]
            
        df = pd.DataFrame(data)
        df['report_year'] = df['report_year'].astype(str)
        # Separate positive and negative samples based on company symbols
        positive_df = df[df['company_symbol'].isin(symbol)]
        negative_df = df[~df['company_symbol'].isin(symbol)]

        # Append the dataframes to the respective lists
        positive_dfs.append(positive_df)
        negative_dfs.append(negative_df)

    # Combine all positive and negative dataframes
    positive_final_df = pd.concat(positive_dfs)
    negative_final_df = pd.concat(negative_dfs)

    # Determine answerability for positive samples based on available years
    positive_final_df['is_answerable'] = positive_final_df['report_year'].isin(available_years)
    negative_final_df['is_answerable'] = False

    # Sample questions based on the specified proportions
    answerable_df = positive_final_df[positive_final_df['is_answerable']]
    unanswerable_df = positive_final_df[~positive_final_df['is_answerable']]
    min_answerable_q_by_year = answerable_df.groupby('report_year').size().min()

    answerable_qa = answerable_df.sample(n = int(q_prop['answerable'] * total_q), replace = False, random_state=random_state)
    unanswerable_qa = unanswerable_df.sample(n = int(q_prop['unanswerable'] * total_q), replace = False, random_state=random_state)
    negative_qa = negative_final_df.sample(n = int(q_prop['negative'] * total_q), replace = False, random_state=random_state)

    final_qa = pd.concat([answerable_qa, unanswerable_qa, negative_qa])
    final_qa.to_csv(output_path, index = False)
