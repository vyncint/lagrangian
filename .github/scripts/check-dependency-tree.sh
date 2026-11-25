#!/usr/bin/env bash
# .github/scripts/check-dependency-tree.sh — the runtime dependency tree is
# exactly what CLAUDE.md's hard rule 2 says it is.
#
#   default features           the crate and nothing else
#   no std, `libm`             the crate and `libm`
#   no std, `libm` + `alloc`   the crate and `libm`
#   every feature              the crate, `libm`, and serde's own tree
#
# "Zero dependencies" is the claim a consumer checks first and the one that
# rots quietest: a convenience crate added behind a default feature compiles,
# passes every test, and makes the README false. Cargo is asked for the
# library's own tree, selected alone (`-p`), normal edges only: with
# `--workspace` another member's feature request is unified into the library,
# and dev-dependencies (`serde_json`, for one round-trip test) are not what a
# consumer builds.
#
# Usage: check-dependency-tree.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

status=0

# `expect <label> <expected crate names, space separated> <cargo tree args ...>`
expect() {
  local label=$1 want=$2
  shift 2
  local got
  got="$(cargo tree -p lagrangian -e normal --prefix none "$@" \
    | sed -E 's/ .*$//' | sort -u | tr '\n' ' ' | sed 's/ $//')"
  if [ "$got" = "$want" ]; then
    echo "dependency tree: $label resolves exactly: $got"
  else
    echo "::error::dependency tree: $label resolves [$got], expected [$want]" >&2
    status=1
  fi
}

expect "the default build" "lagrangian"
expect "the no-std build" "lagrangian libm" --no-default-features --features libm
expect "the no-std build with alloc" "lagrangian libm" --no-default-features --features libm,alloc

# With every feature on, serde is allowed and nothing else is: its own
# dependencies are named here so that a new one is a visible decision.
all="$(cargo tree -p lagrangian -e normal --prefix none --all-features \
  | sed -E 's/ .*$//' | sort -u)"
extra="$(printf '%s\n' "$all" | grep -v -x -E 'lagrangian|libm|serde|serde_core' || true)"
if [ -n "$extra" ]; then
  echo "::error::dependency tree: --all-features resolves crates outside the allowlist (lagrangian, libm, serde, serde_core):" >&2
  printf '%s\n' "$extra" | sed 's/^/  /' >&2
  status=1
else
  echo "dependency tree: --all-features resolves exactly: $(printf '%s' "$all" | tr '\n' ' ')"
fi

exit "$status"
