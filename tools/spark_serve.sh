#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Thin, measured Acer GN100 launcher. Keep policy in the named server profile.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: tools/spark_serve.sh /path/to/model.waste [serve options]" >&2
    exit 2
fi

model=$1
shift
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd -P)

if [[ $(uname -s) != Linux ]]; then
    echo "spark_serve.sh requires Linux" >&2
    exit 2
fi
if [[ ! -d $model ]]; then
    echo "no such WASTE container directory: $model" >&2
    exit 2
fi
for required in "$repo_dir/libwaste.so" "$repo_dir/tools/pm_qos_exec.py"; do
    if [[ ! -f $required ]]; then
        echo "missing $required; run make in $repo_dir first" >&2
        exit 2
    fi
done
for command in python3 taskset curl sudo; do
    if ! command -v "$command" >/dev/null 2>&1; then
        echo "required command is not installed: $command" >&2
        exit 2
    fi
done
if ! sudo -n true; then
    echo "passwordless sudo is required for the request-scoped PM-QoS holder" >&2
    exit 2
fi

cpu_list=${WASTE_SPARK_CPU_LIST:-5-9,15-19}
budget=${WASTE_SPARK_BUDGET:-86583021568}
prefix_cache=${WASTE_SPARK_PREFIX_CACHE:-2G}
prefix_entries=${WASTE_SPARK_PREFIX_ENTRIES:-2}
host=${WASTE_SPARK_HOST:-127.0.0.1}
port=${WASTE_SPARK_PORT:-8000}
max_hold=${WASTE_SPARK_MAX_HOLD_SECONDS:-3600}
startup_timeout=${WASTE_SPARK_STARTUP_TIMEOUT_SECONDS:-900}
artifact_dir=${WASTE_SPARK_QOS_DIR:-/var/tmp/waste-spark-qos-$(id -u)}

# Preserve the qualified head by default, but let a caller select the separate
# semantic-anchor experiment without the wrapper supplying both policies.
prefix_policy=(--conversation-head)
for arg in "$@"; do
    if [[ $arg == --semantic-anchors ]]; then
        prefix_policy=()
        break
    fi
done

# These configure this wrapper, not the engine. The strict profile rejects
# undeclared WASTE_* selectors, so do not let exported launcher knobs leak
# through the holder's deliberately broad WASTE_* allowlist.
unset WASTE_SPARK_CPU_LIST WASTE_SPARK_BUDGET WASTE_SPARK_PREFIX_CACHE
unset WASTE_SPARK_PREFIX_ENTRIES WASTE_SPARK_HOST WASTE_SPARK_PORT
unset WASTE_SPARK_MAX_HOLD_SECONDS WASTE_SPARK_STARTUP_TIMEOUT_SECONDS
unset WASTE_SPARK_QOS_DIR

status_path=$artifact_dir/status.json
events_path=$artifact_dir/events.jsonl

holder_pid=
cleanup() {
    if [[ -n ${holder_pid:-} ]] && kill -0 "$holder_pid" 2>/dev/null; then
        kill -TERM "$holder_pid" 2>/dev/null || true
        wait "$holder_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

echo "Starting Spark server on CPUs $cpu_list with budget $budget"
echo "Q0 evidence: $status_path and $events_path (inspect with sudo)"

sudo -n python3 "$repo_dir/tools/pm_qos_exec.py" \
    --scope requests \
    --max-hold-seconds "$max_hold" \
    --status "$status_path" \
    --events "$events_path" \
    --user "$(id -un)" -- \
    taskset -c "$cpu_list" \
    python3 "$repo_dir/serve/__main__.py" "$model" \
        --host "$host" \
        --port "$port" \
        --budget "$budget" \
        --prefix-cache "$prefix_cache" \
        --prefix-cache-entries "$prefix_entries" \
        "${prefix_policy[@]}" \
        --performance-profile spark-q0 \
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
