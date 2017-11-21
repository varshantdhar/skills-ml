from airflow.hooks import S3Hook
s3_conn = S3Hook().get_conn()
from skills_ml.datasets.job_postings import job_postings, job_postings_chain
from skills_ml.algorithms.corpus_creators.basic import CorpusCreator
from skills_ml.algorithms.sampling.jobs import JobSampler
from skills_utils.time import datetime_to_quarter

import pandas as pd
from collections import Counter
import json

time_range = pd.date_range(start='2011-01-01', freq='Q', periods=24)
time_range = list(map(lambda x: datetime_to_quarter(x), time_range))
print(time_range[0], "to", time_range[-1])
def soc_filter_func(document):
    if document['onet_soc_code']:
        if document['onet_soc_code'][:2] != '99':
            return document

# The weights of whole dataset
weights = {'41': 1522, '29': 619, '35': 600, '53': 426, '11': 367, '15': 263, '43': 204, '13': 159, '49': 135, '25': 120, '17': 100, '33': 89, '51': 82, '37': 67, '31': 53, '19': 50, '21': 41, '47': 31, '27': 28, '39': 27, '45': 14, '23': 3, '55':2}

for key, value in weights.items():
    weights[key] = 1200/value

job_postings_generator = job_postings_chain(s3_conn, time_range, 'open-skills-private/job_postings_common', source='nlx')
corpus = CorpusCreator(job_postings_generator, filter_func=soc_filter_func)
job_sampler = JobSampler(corpus, major_group=True, weights=weights, random_state=42)
corpus_sampled = job_sampler.sample(24000)

with open('samples_24k_v1', 'w') as outfile:
    for c in corpus_sampled:
        outfile.write(json.dumps(c[0]))
        outfile.write('\n')

major_group = list(map(lambda c: c[1][:2], corpus_sampled))
print(len(major_group))
print(Counter(major_group))
