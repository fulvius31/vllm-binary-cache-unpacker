#!/usr/bin/env python3
"""
Convert a *binary* Inductor CompiledArtifact (as saved by vLLM standalone inductor by default)
into the *unpacked* directory form.

Example:
  python vllm_binary_cache_unpack.py \
    --input VLLM_ROOT_CACHE/torch_compile_cache/.../artifact_compile_range_1_8192_subgraph_0 \
    --output ./artifact_compile_range_1_8192_subgraph_0_unpacked \
    --overwrite
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import contextlib


def _rm_rf(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


def extract_artifact_bytes_via_hook(input_file: str) -> bytes:
    """
    Load the CompiledArtifact while intercepting the internal call to
    torch.compiler.load_cache_artifacts(artifact_bytes) so we can capture
    the serialized blob.
    """
    import torch

    if not hasattr(torch, "compiler") or not hasattr(
        torch.compiler, "load_cache_artifacts"
    ):
        raise RuntimeError(
            "This PyTorch does not expose torch.compiler.load_cache_artifacts()."
        )

    if not hasattr(torch, "_inductor") or not hasattr(
        torch._inductor, "CompiledArtifact"
    ):
        raise RuntimeError(
            "This PyTorch does not expose torch._inductor.CompiledArtifact. "
            "This converter targets the standalone Inductor artifact path."
        )

    captured: dict[str, bytes] = {}
    orig = torch.compiler.load_cache_artifacts

    def hooked_load_cache_artifacts(artifact_bytes: bytes, *args, **kwargs):
        # Save the blob; CompiledArtifact.load() will call this
        captured["artifact_bytes"] = artifact_bytes
        return orig(artifact_bytes, *args, **kwargs)

    # Monkeypatch
    torch.compiler.load_cache_artifacts = hooked_load_cache_artifacts  # type: ignore[assignment]
    try:
        # This will parse the binary file and (normally) load cache artifacts.
        # We let it proceed because load() may depend on artifacts being loaded.
        _ = torch._inductor.CompiledArtifact.load(path=input_file, format="binary")
    finally:
        torch.compiler.load_cache_artifacts = orig  # type: ignore[assignment]

    if "artifact_bytes" not in captured:
        raise RuntimeError(
            "Could not capture artifact_bytes. This may be an AOTCompiledArtifact "
            "(which does not use torch.compiler.load_cache_artifacts), or the file "
            "is not a CacheCompiledArtifact binary."
        )

    return captured["artifact_bytes"]


def unpack_to_dir(
    artifact_bytes: bytes, output_dir: str, write_output_code: bool
) -> None:
    import torch
    from torch._inductor.runtime.cache_dir_utils import temporary_cache_dir

    # This context redirects cache dirs so load_cache_artifacts "unpacks" into output_dir
    with temporary_cache_dir(output_dir):
        cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
        if cache_info is None:
            raise RuntimeError(
                "torch.compiler.load_cache_artifacts returned None (unexpected)."
            )

        if not write_output_code:
            return

        # Optional: write human-inspectable generated code files to disk (like Inductor does).
        import pickle
        from torch._inductor.codecache import FxGraphCache

        # cache_info.inductor_artifacts is the set of FxGraphCache keys
        for key in getattr(cache_info, "inductor_artifacts", []):
            subdir = FxGraphCache._get_tmp_dir_for_key(key)  # type: ignore[attr-defined]
            if not os.path.isdir(subdir):
                continue
            for fname in sorted(os.listdir(subdir)):
                fpath = os.path.join(subdir, fname)
                try:
                    with open(fpath, "rb") as f:
                        graph = pickle.load(f)
                    out_file = graph.write_to_disk()
                    print(f"Wrote output code: {out_file}")
                except Exception as e:
                    # Not all entries are guaranteed to be writable/compatible for inspection
                    print(f"NOTE: skipped {fpath}: {e}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert binary vLLM cache artifact to unpacked directory."
    )
    ap.add_argument(
        "--input", required=True, help="Path to the *binary* artifact file."
    )
    ap.add_argument(
        "--output", required=True, help="Output directory for unpacked artifact."
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists.",
    )
    ap.add_argument(
        "--no-output-code",
        action="store_true",
        help="Do not write extra human-inspectable output_code files (faster).",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: --input is not a file: {args.input}", file=sys.stderr)
        return 2

    if os.path.exists(args.output):
        if not args.overwrite:
            print(
                f"ERROR: output exists: {args.output} (use --overwrite)",
                file=sys.stderr,
            )
            return 2
        _rm_rf(args.output)

    os.makedirs(args.output, exist_ok=True)

    try:
        artifact_bytes = extract_artifact_bytes_via_hook(args.input)
        unpack_to_dir(
            artifact_bytes, args.output, write_output_code=not args.no_output_code
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Unpacked artifact written to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
