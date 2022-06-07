from datasets import run_doc_prepro, run_sent_prepro
from utils import parse_opt

if __name__ == '__main__':
    # data_name = 'ag_news'
    data_name = 'yelp_review_full'
    #model_name = 'textcnn'
    model_name = 'han'

    config = parse_opt(data_name, model_name)

    if config.model_name in ['han']:
        run_doc_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            sentence_limit = config.sentence_limit,
            min_word_count = config.min_word_count
        )
    else:
        run_sent_prepro(
            csv_folder = config.dataset_path,
            output_folder = config.output_path,
            word_limit = config.word_limit,
            min_word_count = config.min_word_count
        )
'''
[lijie@db4ai-1 nlp_dl_bench]$ python3 preprocess.py 

Training data: reading and preprocessing...

100%|████████████████████████████████████████████████████████████████████| 120000/120000 [01:11<00:00, 1680.54it/s]

Training data: discarding words with counts less than 5, the size of the vocabulary is 26933.

Training data: word map saved to /ssddisk/data/text_data/outputs/ag_news/docs.

Training data: encoding and padding...

Training data: saving...

Training data: encoded, padded data saved to /ssddisk/data/text_data/outputs/ag_news/docs.

Test data: reading and preprocessing...

100%|████████████████████████████████████████████████████████████████████████| 7600/7600 [00:04<00:00, 1754.40it/s]

Test data: encoding and padding...

Test data: saving...

Test data: encoded, padded data saved to /ssddisk/data/text_data/outputs/ag_news/docs.

All done!
'''