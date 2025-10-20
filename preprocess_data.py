import os
import pandas as pd
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from typing import Dict, List, Any, Optional

RESULTS_COLUMN = "results"     # in the source: a LIST of product IDs per query
WINE_ID_COLUMN = "WineID"        # in the catalog
LABELS_COLUMN = "new_labels"      # one-hot per query (length = #candidates)
QUERY_COLUMN = "query"
FLUSH_EVERY = 2_000                # rows per write_batch
PARQUET_COMPRESSION = "zstd"

class DataPreprocessor:
    def __init__(
        self,
        preprocessing_file: str,   # source parquet with queries, product ID lists, labels
        catalog_file: str,         # parquet with catalog; must contain WineID + product_embed_description
        output_file: str,          # single output parquet path
        batch_size: int = 32,
        max_length: int = 256
    ):
        self.preprocessing_file = preprocessing_file
        self.catalog_dict = self.read_catalog(catalog_file)      # {WineID -> str(product_embed_description)}
        self.output_file = output_file
        self.batch_size = batch_size
        self.max_length = max_length

    
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)

        # Arrow schema: ragged N and ragged token lengths L
        self.schema = pa.schema([
            pa.field("input_ids", pa.list_(pa.list_(pa.int32()))),      # [N, L_i]
            pa.field("attention_mask", pa.list_(pa.list_(pa.int8()))),  # [N, L_i]
            pa.field("label_index", pa.int32()),
            pa.field("n_candidates", pa.int32()),
        ])

    def run_preprocessing(self):
        writer = pq.ParquetWriter(
            self.output_file,
            schema=self.schema,
            compression=PARQUET_COMPRESSION,
            use_dictionary=True,
        )

        buffer = self._empty_buffer()
        try:
            src_pf = pq.ParquetFile(self.preprocessing_file)
            for batch in src_pf.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas() 
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    out = self.preprocess_row(row)
                    if out is None:
                        continue

                    buffer["input_ids"].append(out["input_ids"])
                    buffer["attention_mask"].append(out["attention_mask"])
                    buffer["label_index"].append(out["label_index"])
                    buffer["n_candidates"].append(out["n_candidates"])

                    if len(buffer["input_ids"]) >= FLUSH_EVERY:
                        self._write_buffer_to_parquet(buffer, writer)

            # final flush
            self._write_buffer_to_parquet(buffer, writer)
        finally:
            writer.close()

    def preprocess_row(self, row) -> Optional[Dict[str, Any]]:

        product_ids = row[RESULTS_COLUMN]
        product_texts = [self.catalog_dict.get(pid, "") for pid in product_ids]
        n_products = len(product_texts)

        label_index = np.argmax(row[LABELS_COLUMN])
    
        # tokenize N (query, candidate_i) pairs
        query = row[QUERY_COLUMN]
        enc = self.tokenizer(
            [query] * n_products,
            product_texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,               # store true lengths; do dynamic padding in collate
            return_attention_mask=True, 
        )

        # ensure numeric types for Arrow
        input_ids = [list(map(int, seq)) for seq in enc["input_ids"]]
        attention_mask = [list(map(int, seq)) for seq in enc["attention_mask"]]

        return {
            "input_ids": input_ids,                   # list[list[int]] length N
            "attention_mask": attention_mask,         # list[list[int]] length N
            "label_index": int(label_index),          # scalar
            "n_candidates": int(n_products),          # scalar
        }

    def read_catalog(self, catalog_file: str) -> Dict[Any, str]:
        df = pd.read_parquet(catalog_file)
        # ensure we have the right columns and strings (no NaN)
        if "product_embed_description" not in df.columns:
            raise ValueError("catalog_file must contain 'product_embed_description'.")
        series = (
            df.set_index(WINE_ID_COLUMN)["product_embed_description"]
              .fillna("")
              .astype(str)
        )
        return series.to_dict()

    def _empty_buffer(self) -> Dict[str, List[Any]]:
        return {
            "input_ids": [],
            "attention_mask": [],
            "label_index": [],
            "n_candidates": [],
        }

    def _write_buffer_to_parquet(self, buffer: Dict[str, List[Any]], writer: pq.ParquetWriter):
        if not buffer["input_ids"]:
            return
        arrays = {
            "input_ids": pa.array(buffer["input_ids"], type=self.schema.field("input_ids").type),
            "attention_mask": pa.array(buffer["attention_mask"], type=self.schema.field("attention_mask").type),
            "label_index": pa.array(buffer["label_index"], type=self.schema.field("label_index").type),
            "n_candidates": pa.array(buffer["n_candidates"], type=self.schema.field("n_candidates").type),
        }
        batch = pa.record_batch(arrays, schema=self.schema)
        writer.write_batch(batch)
        # clear buffer
        for k in buffer:
            buffer[k].clear()


if __name__ == "__main__":

    train_click = "click_train.parquet"
    test_click = "click_test.parquet"
    catalog = "catalog.parquet"
    output_train = "preprocessed_click_train.parquet"
    output_test = "preprocessed_click_test.parquet"
    preprocessor = DataPreprocessor(
        preprocessing_file=train_click,
        catalog_file=catalog,
        output_file=output_train,
        batch_size=64,
        max_length=256
    )
    preprocessor.run_preprocessing()
    preprocessor = DataPreprocessor(
        preprocessing_file=test_click,
        catalog_file=catalog,
        output_file=output_test,
        batch_size=64,
        max_length=256
    )
    preprocessor.run_preprocessing()
