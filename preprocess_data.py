import pandas as pd
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

ID_COLUMN = 'WineID'
SCHEMA = pa.schema([       
    pa.field("input_ids", pa.list_(pa.list_(pa.int32()))),  # [N, L]
    pa.field("attention_mask", pa.list_(pa.list_(pa.int8()))),
    pa.field("label_index", pa.int32()),
    pa.field("n_candidates", pa.int32()),
])


class DataPreprocessor:
    def __init__(self, preprocessing_file: str, catalog_file: str, output_file: str, batch_size: int = 32, max_length: int = 256, device: str = None):
        self.preprocessing_file = preprocessing_file
        self.catalog_dict = self.read_catalog(catalog_file)
        self.output_file = output_file
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", device=self.device)
        self.schema = self._init_output_schema()
        # This is where we will store rows before writing to Parquet
        self.buffer = self._empty_buffer()

    def _empty_buffer(self):
        return {
            "query_id": [],
            "input_ids": [],
            "attention_mask": [],
            "label_index": [],
            "n_candidates": [],
        }

    def _init_output_schema(self):
        self.schema = pa.schema([
        pa.field("input_ids", pa.list_(pa.list_(pa.int32()))),  # [N, L]
        pa.field("attention_mask", pa.list_(pa.list_(pa.int8()))),
        pa.field("label_index", pa.int32()),
        pa.field("n_candidates", pa.int32()),
    ])


    def run_preprocessing(self):
        parquet_file = pq.ParquetFile(self.preprocessing_file)
        processed_rows = []
        for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            df = batch.to_pandas()
            for idx in range(len(df)):
                row = df.iloc[idx]
                tokenized =  self._process_row(row)
                processed_rows.append({
                    'input_ids': tokenized['input_ids'],  # List of lists (variable length)
                    'attention_mask': tokenized['attention_mask'],
                    'labels': np.argmax(row['labels']),
                    'n_products': len(row['labels']) # I am not assuming labels are the same size, but 
                })
    

    def preprocess_row(self, row):
        product_results = row[ID_COLUMN]
        product_results = [self.catalog_dict.get(product_results[i], {}) for i in range(len(product_results))]
        n_products= len(product_results)
        # I have decided to not pad this data here, but rather in the model forward pass
        tokenized_row = self.tokenizer([row['query']]*n_products,product_results , 
                                       padding=False, 
                                       truncation=True, 
                                       max_length=self.max_length, 
                                       return_tensors=None)
        return tokenized_row

    def read_catalog(self, catalog_file):
        df_catalog = pd.read_parquet(catalog_file)
        return df_catalog.set_index(ID_COLUMN)['product_embed_description'].to_dict()


    def save_preprocessed_rows(self, data, output_directory):
        pass


    
 


    def _write_record_batch(self, writer, record_batch):
        pass


    def flush_rows_to_shard(self, rows, shard_idx_ref):
        pass




