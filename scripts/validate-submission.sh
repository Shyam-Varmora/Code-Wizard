#!/usr/bin/env bash
# Copied from hackathon materials: ping HF Space /reset, docker build, openenv validate.
# Usage: ./scripts/validate-submission.sh https://your-space.hf.space /path/to/ticket-router-env

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %s.${NC}\n" "$1"
  exit 1
}

printf "\n${BOLD}OpenEnv Submission Validator${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"

log "${BOLD}Step 1/3: HF Space /reset${NC}"
CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space responds to POST /reset"
else
  fail "HTTP $HTTP_CODE (expected 200)"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: docker build${NC}"
if ! command -v docker &>/dev/null; then
  fail "docker not found"
  stop_at "Step 2"
fi
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile in repo root or server/"
  stop_at "Step 2"
fi
BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t ticket-router-validate "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true
if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: openenv validate${NC}"
if ! command -v openenv &>/dev/null; then
  fail "openenv not found (pip install openenv-core)"
  stop_at "Step 3"
fi
if (cd "$REPO_DIR" && openenv validate 2>&1); then
  pass "openenv validate"
else
  stop_at "Step 3"
fi

printf "\n${GREEN}${BOLD}All checks passed.${NC}\n\n"
