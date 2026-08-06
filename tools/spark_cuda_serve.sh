#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Qualified single-user GB10 CUDA server with child-scoped Q0.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: tools/spark_cuda_serve.sh /path/to/model.waste [serve options]" >&2
    exit 2
fi

model=$1
shift
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)

if [[ $(uname -s) != Linux ]]; then
    echo "spark_cuda_serve.sh requires Linux" >&2
    exit 2
fi
if [[ ! -d $model ]]; then
    echo "no such WASTE container directory: $model" >&2
    exit 2
fi
for required in "$repo_dir/libwaste.so" "$repo_dir/tools/pm_qos_exec.py"; do
    if [[ ! -f $required ]]; then
        echo "missing $required; build with WASTE_ENABLE_CUDA=1 first" >&2
        exit 2
    fi
done
for command in python3 taskset curl sudo; do
    command -v "$command" >/dev/null 2>&1 || {
        echo "required command is not installed: $command" >&2
        exit 2
    }
done
sudo -n true || {
    echo "passwordless sudo is required for child-scoped PM-QoS" >&2
    exit 2
}

cpu_list=${WASTE_SPARK_CUDA_CPU_LIST:-5-9,15-19}
# 59,340 MiB expert cache plus K3's measured floor and a 2 GiB prefix reserve.
budget=${WASTE_SPARK_CUDA_BUDGET:-95172120576}
prefix_cache=${WASTE_SPARK_CUDA_PREFIX_CACHE:-2G}
prefix_entries=${WASTE_SPARK_CUDA_PREFIX_ENTRIES:-2}
host=${WASTE_SPARK_CUDA_HOST:-127.0.0.1}
port=${WASTE_SPARK_CUDA_PORT:-8000}
startup_timeout=${WASTE_SPARK_CUDA_STARTUP_TIMEOUT_SECONDS:-900}
status_path=${WASTE_SPARK_CUDA_QOS_STATUS:-/var/lib/waste-qos/spark-cuda-status.json}

unset WASTE_SPARK_CUDA_CPU_LIST WASTE_SPARK_CUDA_BUDGET
unset WASTE_SPARK_CUDA_PREFIX_CACHE WASTE_SPARK_CUDA_PREFIX_ENTRIES
unset WASTE_SPARK_CUDA_HOST WASTE_SPARK_CUDA_PORT
unset WASTE_SPARK_CUDA_STARTUP_TIMEOUT_SECONDS WASTE_SPARK_CUDA_QOS_STATUS

holder_pid=
cleanup() {
    if [[ -n ${holder_pid:-} ]] && kill -0 "$holder_pid" 2>/dev/null; then
        kill -TERM "$holder_pid" 2>/dev/null || true
        wait "$holder_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

echo "Starting qualified Spark CUDA server on CPUs $cpu_list"
echo "Q0 evidence: $status_path"

sudo -n python3 "$repo_dir/tools/pm_qos_exec.py" \
    --scope child \
    --latency-us 0 \
    --status "$status_path" \
    --user "$(id -un)" -- \
    taskset -c "$cpu_list" \
    python3 "$repo_dir/serve/__main__.py" "$model" \
        --host "$host" \
        --port "$port" \
        --budget "$budget" \
        --prefix-cache "$prefix_cache" \
        --prefix-cache-entries "$prefix_entries" \
        --conversation-head \
        --performance-profile spark-cuda \
        "$@" &
holder_pid=$!

deadline=$((SECONDS + startup_timeout))
health_url="http://$host:$port/health"
while ! curl --fail --silent --show-error --max-time 2 \
        "$health_url" >/dev/null 2>&1; do
    if ! kill -0 "$holder_pid" 2>/dev/null; then
        wait "$holder_pid"
        exit $?
    fi
    if ((SECONDS >= deadline)); then
        echo "server did not become healthy within ${startup_timeout}s" >&2
        exit 1
    fi
    sleep 2
done

echo "Ready: $health_url"
wait "$holder_pid"
