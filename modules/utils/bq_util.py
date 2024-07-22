#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Thu Mar 18 18:24:38 2021

# @author: iizmotabar

from google.cloud import  bigquery
import json
import pandas as pd

def upload_df_to_bq(dataframe,bq_client,table_name, dataset_id='test_dataset', project_id='default', write_disposition='WRITE_APPEND'):
    '''
    this function uploads data into defined table.
    parameter: data frame and bigquery client
    returns: job object
    '''
    try:
        print('upload starting')
        table_id = f"{project_id}.{dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig()
        job = bigquery_data_load_job(dataframe, bq_client, job_config, table_id)
        print(job.result())
        print(f'Success no of rows updated: {job.output_rows}')
        return job
    except Exception as e:
        error = e
        print('Failed to upload ---- ')
        print(f'error: {error}')
        return None

def bigquery_data_load_job(dataframe, bq_client, job_config, table_id):
    try:
        job = bq_client.load_table_from_dataframe(
                dataframe, table_id, job_config=job_config
            )
        return job
    except Exception as e:
        raise Exception(e.args[0])
        
def run_query(script,client):
    '''
    runs query into bigquery and returns results as dataframe
    '''
    try:
        print("start ------ run_query()")
        #bq client
        print('--------------------------')
        print(script)
        print('--------------------------')
        results = client.query(script).result()
        lis = [dict(i) for i in results]
        pj = json.loads(json.dumps(lis))
        results_df = pd.DataFrame(pj)
        if all(results_df.shape):
            return results_df
#        results_df = results_df.fillna(0)
        print('success run_query with shape:', results_df.shape)
        return None
    except Exception as e:
        print('unsuccess run_query')
        print('error:' + str(e))