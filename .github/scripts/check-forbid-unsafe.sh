#!/usr/bin/env bash
# .github/scripts/check-forbid-unsafe.sh — SECURITY.md says the published
# crate contains no `unsafe`; this holds the sentence to the code.
#
# Two halves, because either alone can be true while the claim is false:
#
#   * the crate root carries `#![forbid(unsafe_code)]`, which no inner
#     `#[allow]` can override — remove the line and every module is free to
#     add a block that the workspace's `unsafe_code = "warn"` would let a
#     single `#[allow(unsafe_code)]` silence;
#   * SECURITY.md still makes the statement this script is named for, in the
#     words below, so the document cannot quietly stop promising it.
#
# Usage: .github/scripts/check-forbid-unsafe.sh [SECURITY.md]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

doc="${1:-SECURITY.md}"
lib="crates/lagrangian/src/lib.rs"
status=0

if ! grep -Eq '^#!\[forbid\(unsafe_code\)\]' "$lib"; then
  echo "::error::$lib does not carry #![forbid(unsafe_code)]" >&2
  status=1
fi

if [ ! -f "$doc" ] || ! grep -Fq '**The crate contains no `unsafe`.**' "$doc"; then
  echo "::error::$doc does not carry the statement this gate holds (expected: '**The crate contains no \`unsafe\`.**')" >&2
  status=1
fi

# `forbid` already makes a block a compile error; this catches the word in a
# place the lint does not look, such as a macro body that is never expanded.
hits="$(grep -rn -E '(^|[^a-zA-Z_])unsafe[[:space:]]*(\{|fn|impl|trait|extern)' crates/lagrangian/src || true)"
if [ -n "$hits" ]; then
  echo "::error::crates/lagrangian/src uses unsafe:" >&2
  printf '%s\n' "$hits" | sed 's/^/  /' >&2
  status=1
fi

if [ "$status" -eq 0 ]; then
  echo "unsafe: $lib forbids it, $doc says so, and crates/lagrangian/src contains none"
fi
exit "$status"
