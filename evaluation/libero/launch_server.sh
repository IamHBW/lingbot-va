model_path=${1:?Usage: $0 MODEL_PATH [SAVE_ROOT]}

save_root=${2:-visualization/}
mkdir -p "$save_root"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port 29056 \
    --model-path "$model_path" \
    --save_root "$save_root"
