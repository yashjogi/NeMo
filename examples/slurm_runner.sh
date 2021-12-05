set -e -x

sbatch_command="$1"
results_dir="$2"

output=$(eval ${sbatch_command})
job_id=$(echo ${output} | grep -oP "\d+")
echo "Current job id: ${job_id}"

sleep 5

while [ ! -z "$(squeue | grep "${job_id}")" ]; do
  sleep 600
done

path_to_errors="$(find "${results_dir}" -name "error-${job_id}-0.out")"

while [ ! -z "$(grep 'DUE TO TIME LIMIT ***' "${path_to_errors}")" ]; do
  echo RESTARTING
  output=$(eval ${sbatch_command})
  job_id=$(echo ${output} | grep -oP "\d+")
  echo "Current job id: ${job_id}"

  sleep 5

  while [ ! -z "$(squeue | grep "${job_id}")" ]; do
    sleep 600
  done

  path_to_errors="$(find "${results_dir}" -name "error-${job_id}-0.out")"
done

set +e +x