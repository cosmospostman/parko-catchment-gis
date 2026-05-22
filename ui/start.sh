#!/usr/bin/env bash
set -euo pipefail

cleanup() {
    echo "Shutting down..."
    kill "$serve_pid" "$dev_pid" 2>/dev/null
    wait "$serve_pid" "$dev_pid" 2>/dev/null
    exit 0
}

trap cleanup INT TERM

restart_loop() {
    local name="$1"
    shift
    while true; do
        "$@" &
        local pid=$!
        wait "$pid" && true
        local code=$?
        [[ $code -eq 0 || $code -eq 130 || $code -eq 143 ]] && break
        echo "$name exited with code $code, restarting..."
    done
}

restart_loop "serve" deno task serve &
serve_pid=$!

restart_loop "dev" deno task dev &
dev_pid=$!

wait "$serve_pid" "$dev_pid"
