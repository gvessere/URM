run_name="URM-arcagi1"
checkpoint_path="checkpoints/${run_name}" 
mkdir -p $checkpoint_path

if [ "$(uname -s)" = "Darwin" ]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
fi

torchrun --nproc-per-node "${NPROC_PER_NODE}" pretrain.py \
data_path=data/arc1concept-aug-1000 \
arch=urm arch.loops=16 arch.H_cycles=2 arch.L_cycles=6 arch.num_layers=4 \
epochs=200000 \
eval_interval=2000 \
puzzle_emb_lr=1e-2 \
weight_decay=0.1 \
+run_name=$run_name \
+checkpoint_path=$checkpoint_path \
+ema=True
