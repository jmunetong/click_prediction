import torch
from torch.utils.data import IterableDataset, DataLoader
import pyarrow.parquet as pq
import pandas as pd
import random
from collections import deque

class RelatedParquetStreamDataset(IterableDataset):
    def __init__(self, main_file: str, catalog_file: str, id_column: str = 'results', 
                 batch_size: int = 32, buffer_size: int = 1000, shuffle: bool = True, 
                 seed: int = None, prefill_buffer: bool = True):
        """
        Args:
            main_file: Path to main parquet file
            catalog_file: Path to catalog parquet file with product info
            id_column: Column name for the foreign key
            batch_size: Number of rows to read at once from main file
            buffer_size: Size of the shuffle buffer
            shuffle: Whether to shuffle samples using the buffer
            seed: Random seed for reproducibility
            prefill_buffer: If True, fill buffer completely before yielding
        """
        self.main_file = main_file
        self.id_column = id_column
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        self.prefill_buffer = prefill_buffer
        
        self.catalog_data = self._load_catalog_data(catalog_file)
    
    def _load_catalog_data(self, catalog_file: str) -> dict:
        """Load catalog data parquet into a dictionary for fast lookups"""
        df = pd.read_parquet(catalog_file)
        return df.set_index('WineID')['product_embed_description'].to_dict()
    
    def _process_row(self, row):
        """Process a single row and return the sample dict"""
        product_results = row[self.id_column].values[0]
        product_results = [self.catalog_data.get(product_results[i], {}) for i in range(len(product_results))]
        query = row['query']
        labels = row['new_labels']
        return {
            'query': query,
            'product_results': product_results,
            'labels': torch.tensor(labels, dtype=torch.float32)
        }
    
    def __iter__(self):
        # Set random seed
        worker_info = torch.utils.data.get_worker_info()
        if self.seed is not None:
            seed = self.seed + (worker_info.id if worker_info else 0)
            random.seed(seed)
        
        if not self.shuffle:
            # No shuffling
            parquet_file = pq.ParquetFile(self.main_file)
            for batch in parquet_file.iter_batches(batch_size=self.batch_size):
                df = batch.to_pandas()
                for idx in range(len(df)):
                    yield self._process_row(df.iloc[idx])
        else:
            # Shuffle using buffer (deque for efficient operations)
            buffer = deque(maxlen=self.buffer_size)
            parquet_file = pq.ParquetFile(self.main_file)
            
            batch_iterator = parquet_file.iter_batches(batch_size=self.batch_size)
            
            # Prefill buffer if requested
            if self.prefill_buffer:
                samples_added = 0
                for batch in batch_iterator:
                    df = batch.to_pandas()
                    for idx in range(len(df)):
                        buffer.append(self._process_row(df.iloc[idx]))
                        samples_added += 1
                        if samples_added >= self.buffer_size:
                            break
                    if samples_added >= self.buffer_size:
                        break
            
            # Continue reading and yielding
            for batch in batch_iterator:
                df = batch.to_pandas()
                
                for idx in range(len(df)):
                    sample = self._process_row(df.iloc[idx])
                    
                    if len(buffer) >= self.buffer_size:
                        # Buffer is full, yield a random sample
                        random_idx = random.randint(0, len(buffer) - 1)
                        yield buffer[random_idx]
                        buffer[random_idx] = sample  # Replace with new sample
                    else:
                        # Still filling buffer
                        buffer.append(sample)
            
            # Shuffle and yield remaining buffer
            buffer_list = list(buffer)
            random.shuffle(buffer_list)
            for sample in buffer_list:
                yield sample

# Usage examples
dataset = RelatedParquetStreamDataset(
    main_file='transactions.parquet',
    catalog_file='products.parquet',
    id_column='product_id',
    buffer_size=1000,
    shuffle=True,
    seed=42,
    prefill_buffer=True  # Better randomness at start
)

dataloader = DataLoader(dataset, batch_size=32, num_workers=4)