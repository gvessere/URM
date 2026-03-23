run_name="URM-sudoku"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

# Same default as README (build_sudoku_dataset.py --output-dir ...). Override: DATA_PATH=/path/to/data sh scripts/URM_sudoku.sh
DATA_PATH="${DATA_PATH:-data/sudoku-extreme-1k-aug-1000}"
if [ ! -d "${DATA_PATH}/train" ]; then
  echo "Dataset not found: ${DATA_PATH}/train" >&2
  echo "Build it (from repo root), for example:" >&2
  echo "  python data/build_sudoku_dataset.py --output-dir ${DATA_PATH} --subsample-size 1000 --num-aug 1000" >&2
  echo "Or point DATA_PATH at an existing dataset that contains train/ and test/ splits." >&2
  exit 1
fi

# Linux + CUDA: use NPROC_PER_NODE=8 (or your GPU count). macOS / CPU-only PyTorch has no NCCL;
# pretrain.py uses gloo on CPU. Default to 1 process on Darwin; override with NPROC_PER_NODE.
if [ "$(uname -s)" = "Darwin" ]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi

torchrun --nproc-per-node "${NPROC_PER_NODE}" pretrain.py \
data_path="${DATA_PATH}" \
arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
epochs=50000 \
eval_interval=2000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
