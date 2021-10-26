#!/bin/bash
#SBATCH -A ent_aiapps_asr
#SBATCH -p batch                 # luna / backfill / interactive
#SBATCH -N 1                    # number of nodes
#SBATCH -t 8:00:00              # wall time  (4 for luna, 8 for backfill, 2 for interactive)
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --gpus-per-node=16
#SBATCH -J "ent_aiapps_asr:punctuation_capitalization_on_wiki_with_quotes"  # job name (<< CHANGE ! >>)
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --overcommit
#SBATCH --ntasks-per-node=16     # n tasks per machine (one task per gpu) <required>
set -x
SLURM_ACCOUNT_DIR='swdl/swdl-langspeech'  # <Make sure you dont override SLURM_ACCOUNT!>
USERID='apeganov'
CONTAINER="gitlab-master.nvidia.com:5005/apeganov/speechtranslation:latest"
WANDB="${wandb}" # replace with your own WandB API key

# Training - we want to train for 300B tokens with a global batch size of at least 1M tokens
# total_tokens = max_steps * global_batch_size_in_tokens
# global_batch_size_in_tokens = micro_batch_size * data_parallel_size * accumulate_grad_batches * seq_length
# data_parallel_size = num_nodes * num_gpus_per_node (no model parallel)
VAL_CHECK_INTERVAL=2000
LOG_EVERY_N_STEPS=100

# Logging
PROJECT="autoregressive_punctuation_capitalization"
EXPNAME="evelina_wiki_with_quotes_draco"

# Mounts
SLURM_ACCOUNT='ent_aiapps'
USERID='apeganov'
LUSTRE_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}
DATA="${LUSTRE_ACCOUNT_PREFIX}/datasets/data/punctuation_capitalization/wiki_with_quotes_48_65"
RESULTS=${LUSTRE_ACCOUNT_PREFIX}/users/${USERID}/results/$PROJECT/$EXPNAME
CODE="${LUSTRE_ACCOUNT_PREFIX}/users/${USERID}/code/NeMo"

mkdir -p ${RESULTS}

MOUNTS="--container-mounts=$CODE:/code,$RESULTS:/results,$DATA:/data"

# Necessary Exports
export HYDRA_FULL_ERROR=1

OUTFILE="${RESULTS}/slurm-%j-%n.out"
ERRFILE="${RESULTS}/error-%j-%n.out"

read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& wandb login ${WANDB} \
&& echo "Starting training" \
&& cd /code/ \
&& CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 python \
  /code/examples/nlp/token_classification/punctuation_capitalization_train.py \
	--config-path=/code/examples/nlp/token_classification/conf/wiki \
	--config-name=train_local \
	model.train_ds.metadata_file="/data/train_tarred/metadata.punctuation_capitalization.tokens15000.max_seq_length512.bert-base-uncased.json" \
	model.validation_ds.text_file="/data/dev/input.txt" \
	model.validation_ds.labels_file="/data/dev/bert_labels.txt" \
	model.test_ds.text_file="/data/test/input.txt" \
	model.test_ds.labels_file="/data/test/bert_labels.txt" \
	model.class_labels.punct_labels_file="/data/train_tarred/punct_label_ids.csv" \
	model.class_labels.capit_labels_file="/data/train_tarred/capit_label_ids.csv" \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
	trainer.gpus=${SLURM_NTASKS_PER_NODE} \
	trainer.max_epochs=null \
	trainer.max_steps=${MAX_STEPS} \
	trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	exp_manager.explicit_log_dir=/results \
	exp_manager.resume_if_exists=True \
	exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=val_loss \
	exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	exp_manager.checkpoint_callback_params.always_save_nemo=False
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
set +x
