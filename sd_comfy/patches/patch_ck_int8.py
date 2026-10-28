"""Fall back off comfy_kitchen's INT8 attention kernels on old drivers.

Models such as MOSS-Image (`*_int8_convrot.safetensors`) carry a per-layer
state-dict config `{"attention": "comfy_kitchen_int8"}`. comfy_kitchen's CUDA
extension is built against CUDA 13 and needs driver >= ~580; Paperspace Free
machines ship 550, so sampling dies with:

    RuntimeError: detect_k_anchor kernel launch failed:
    CUDA driver version is insufficient for CUDA runtime version

This patch wraps that branch with a COMFY_DISABLE_CK_INT8 env guard (set to 1
in main.sh). With the guard on, the layer keeps function=None and ComfyUI uses
its normal attention path instead. Idempotent: re-run after every git pull.
"""
import pathlib
import shutil
import sys

ATTENTION = pathlib.Path('/storage/stable-diffusion-comfy/comfy/ldm/modules/attention.py')

OLD = 'if COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE and comfy_kitchen.int8_attention_is_available('
NEW = ('if not __import__("os").environ.get("COMFY_DISABLE_CK_INT8") and '
       'COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE and comfy_kitchen.int8_attention_is_available(')


def main():
    if not ATTENTION.exists():
        print('attention.py not found')
        return 1
    src = ATTENTION.read_text()
    if 'COMFY_DISABLE_CK_INT8' in src:
        print('already patched')
        return 0
    if src.count(OLD) != 1:
        print('unexpected match count:', src.count(OLD))
        return 1
    backup = pathlib.Path('/storage/backups')
    backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ATTENTION, backup / 'attention.py.orig')
    ATTENTION.write_text(src.replace(OLD, NEW, 1))
    print('patched')
    return 0


if __name__ == '__main__':
    sys.exit(main())
