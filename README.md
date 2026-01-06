# vLLM Binary Cache Unpacker

vLLM’s compiled cache artifacts are now saved **as a single binary file by default** (standalone Inductor artifact). This tool converts an **already-existing vLLM binary cache artifact** into an **unpacked directory layout**, so you can **inspect** what’s inside (cache contents / generated code / metadata) without re-compiling.

## What it does
- Input: a vLLM binary cache artifact file (e.g. `.../torch_compile_cache/.../artifact_compile_range_...`)
- Output: an “unpacked” folder containing the restored cache files (and optionally extra human-readable code dumps)

## Usage

```bash
python vllm_binary_cache_unpack.py \
  --input /path/to/vllm_binary_artifact \
  --output ./artifact_unpacked \
  --overwrite
````

Optional (faster, skips extra code dumps):

```bash
python vllm_binary_cache_unpack.py \
  --input /path/to/vllm_binary_artifact \
  --output ./artifact_unpacked \
  --overwrite \
  --no-output-code
```

## Notes

* Run this with the **same PyTorch build/version** used to produce the cache, otherwise loading may fail due to mismatched cache keys.
* This is intended for **inspection/debugging** of an existing cache; it does not create new compiled artifacts.

