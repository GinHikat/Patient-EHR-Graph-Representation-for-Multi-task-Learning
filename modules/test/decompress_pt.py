import torch
import numpy as np
import json
import time
import pickle
import zipfile
import concurrent.futures
import os
from pathlib import Path
from tqdm import tqdm

class MockStorage:
    def __init__(self, dtype, key, numel, storage_type=None):
        self.dtype = dtype
        self.storage_key = key
        self.numel = numel
        self.device = torch.device('cpu')
        
        self._untyped_storage = None
        if hasattr(torch, 'UntypedStorage'):
            try:
                self._untyped_storage = torch.UntypedStorage()
            except Exception:
                pass
        
        if self._untyped_storage is None and hasattr(torch, 'storage') and hasattr(torch.storage, '_UntypedStorage'):
            try:
                self._untyped_storage = torch.storage._UntypedStorage()
            except Exception:
                pass
                
        if self._untyped_storage is None and storage_type is not None:
            try:
                self._untyped_storage = storage_type()
            except Exception:
                pass
                
        if self._untyped_storage is not None:
            try:
                self._untyped_storage.storage_key = key
                self._untyped_storage.numel = numel
            except Exception:
                pass

    def _untyped(self):
        return self._untyped_storage if self._untyped_storage is not None else self

TENSOR_STORAGE_MAP = {}

class SimpleUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_keys = []

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5:
            typename, storage_type, key, location, numel = pid[:5]
            self.storage_keys.append(key)
            dtype = getattr(storage_type, 'dtype', torch.float32)
            return MockStorage(dtype, key, numel, storage_type)
        return pid

    def find_class(self, module, name):
        import torch._utils
        if module == 'torch._utils' and name.startswith('_rebuild_tensor'):
            orig_func = getattr(torch._utils, name)
            def wrapped_rebuild(storage, *args, **kwargs):
                tensor = orig_func(storage, *args, **kwargs)
                if isinstance(storage, MockStorage):
                    TENSOR_STORAGE_MAP[id(tensor)] = storage
                return tensor
            return wrapped_rebuild
        return super().find_class(module, name)

def save_patient(pid, emb_bytes, emb_shape, emb_dtype, dt_bytes, dt_shape, dt_dtype, meta, output_dir):
    # Convert bytes to numpy arrays
    emb = np.frombuffer(emb_bytes, dtype=emb_dtype).reshape(emb_shape)
    dt = np.frombuffer(dt_bytes, dtype=dt_dtype).reshape(dt_shape)
    
    # Paths
    emb_path = output_dir / f'{pid}_emb.npy'
    dt_path = output_dir / f'{pid}_dt.npy'
    meta_path = output_dir / f'{pid}_meta.json'
    
    # Save files
    np.save(emb_path, emb)
    np.save(dt_path, dt)
    
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
        
    return emb.nbytes / (1024 ** 2)

def main():
    input_path = './data/patient_timelines_new.pt'
    output_dir = Path('./data/Timeline_new')
    start_total = time.time()
    
    print('=' * 60)
    print('Loading .pt file structure streamingly...')
    start_load = time.time()
    
    # 1. Open the zip file
    if not zipfile.is_zipfile(input_path):
        raise ValueError(f"File {input_path} is not a valid zip file or .pt archive.")
        
    z = zipfile.ZipFile(input_path, 'r')
    
    # 2. Find and unpickle the data.pkl file
    pkl_name = [name for name in z.namelist() if name.endswith('data.pkl')][0]
    with z.open(pkl_name) as f:
        unpickler = SimpleUnpickler(f)
        data = unpickler.load()
        storage_keys = getattr(unpickler, 'storage_keys', [])
        
    load_time = time.time() - start_load
    print(f'Loaded structure of {len(data):,} patients in {load_time:.2f}s')
    
    print('\nCreating output directory...')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output directory: {output_dir.resolve()}')
    
    print('\nStarting parallel extraction...\n')
    
    num_patients = len(data)
    total_emb_size_mb = 0
    failed_patients = []
    
    # Mapping torch dtypes to numpy dtypes
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_
    }
    
    # Configure thread pool
    max_workers = min(32, (os.cpu_count() or 1) + 4)
    max_queued_tasks = max_workers * 2
    
    print(f"Using ThreadPoolExecutor with {max_workers} workers.")
    
    active_futures = set()
    
    # We will use the thread pool executor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, (pid, content) in enumerate(
            tqdm(
                data.items(),
                total=num_patients,
                desc='Extracting patients',
                unit='patient',
                dynamic_ncols=True
            ),
            start=1
        ):
            try:
                # Retrieve tensors
                emb_tensor = content['emb']
                dt_tensor = content['dt']
                
                # Get storage keys
                emb_key = None
                dt_key = None
                
                # 1. Try to get storage key via custom unpickling registry (100% robust bypass)
                emb_storage = TENSOR_STORAGE_MAP.get(id(emb_tensor))
                dt_storage = TENSOR_STORAGE_MAP.get(id(dt_tensor))
                if emb_storage is not None:
                    emb_key = emb_storage.storage_key
                if dt_storage is not None:
                    dt_key = dt_storage.storage_key
                
                # 2. Fallback to sequential list if registry lookup failed
                if (emb_key is None or dt_key is None) and len(storage_keys) == 2 * num_patients:
                    try:
                        emb_key = storage_keys[2 * (idx - 1)]
                        dt_key = storage_keys[2 * (idx - 1) + 1]
                    except Exception:
                        pass
                
                # 3. Last resort fallback to extracting from tensor storage
                if emb_key is None or dt_key is None:
                    try:
                        emb_storage_obj = emb_tensor.untyped_storage() if hasattr(emb_tensor, 'untyped_storage') else emb_tensor.storage()
                        emb_key = getattr(emb_storage_obj, 'storage_key', None) or getattr(emb_storage_obj, '_key', None)
                    except Exception:
                        pass
                    try:
                        dt_storage_obj = dt_tensor.untyped_storage() if hasattr(dt_tensor, 'untyped_storage') else dt_tensor.storage()
                        dt_key = getattr(dt_storage_obj, 'storage_key', None) or getattr(dt_storage_obj, '_key', None)
                    except Exception:
                        pass
                
                # Read bytes from zip streamingly
                emb_bytes = z.read(f'patient_timelines/data/{emb_key}')
                dt_bytes = z.read(f'patient_timelines/data/{dt_key}')
                
                # Get shapes & numpy dtypes
                emb_shape = tuple(emb_tensor.shape)
                emb_dtype = dtype_map.get(emb_tensor.dtype, np.float32)
                
                dt_shape = tuple(dt_tensor.shape)
                dt_dtype = dtype_map.get(dt_tensor.dtype, np.float32)
                
                # Submit task
                fut = executor.submit(
                    save_patient,
                    pid,
                    emb_bytes,
                    emb_shape,
                    emb_dtype,
                    dt_bytes,
                    dt_shape,
                    dt_dtype,
                    content['meta'],
                    output_dir
                )
                fut.pid = pid
                active_futures.add(fut)
                
                # Keep memory usage constant by waiting if queue is full
                while len(active_futures) >= max_queued_tasks:
                    done, active_futures = concurrent.futures.wait(
                        active_futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for f_done in done:
                        try:
                            total_emb_size_mb += f_done.result()
                        except Exception as e:
                            failed_patients.append((f_done.pid, str(e)))
                            tqdm.write(f'[ERROR] Failed patient {f_done.pid}: {e}')
                            
            except Exception as e:
                failed_patients.append((pid, str(e)))
                tqdm.write(f'[ERROR] Failed patient {pid}: {e}')
                
        # Wait for all remaining futures to complete
        if active_futures:
            done, _ = concurrent.futures.wait(active_futures)
            for f_done in done:
                try:
                    total_emb_size_mb += f_done.result()
                except Exception as e:
                    failed_patients.append((f_done.pid, str(e)))
                    tqdm.write(f'[ERROR] Failed patient {f_done.pid}: {e}')
                    
    z.close()
    
    total_time = time.time() - start_total
    
    print('\n' + '=' * 60)
    print('EXTRACTION COMPLETE')
    print('=' * 60)
    
    print(f'Total patients processed : {num_patients:,}')
    print(f'Successful               : {num_patients - len(failed_patients):,}')
    print(f'Failed                   : {len(failed_patients):,}')
    print(f'Total embedding storage  : {total_emb_size_mb:.2f} MB')
    print(f'Total runtime            : {total_time:.2f}s')
    
    if failed_patients:
        print('\nFailed patient IDs:')
        for pid, err in failed_patients[:10]:
            print(f' - {pid}: {err}')
            
        if len(failed_patients) > 10:
            print(f'... and {len(failed_patients) - 10} more')

if __name__ == '__main__':
    main()