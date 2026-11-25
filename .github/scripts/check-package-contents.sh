#!/usr/bin/env bash
# .github/scripts/check-package-contents.sh — the published crate carries
# what it must and nothing it should not.
#
# Must: `README.md`, `LICENSE-MIT` and `LICENSE-APACHE`. They live at the
# repository root and reach the package through three symlinks in
# crates/lagrangian/; a checkout that lost them (a Windows clone without
# symlink support, a tool that replaced one with its target's path as text)
# still builds and still passes every test, and would publish a crate with no
# license text and a broken README.
#
# Must not: anything outside the allowlist below. The planning documents at
# the repository root (PLAN.md, PROMPT.md, CLAUDE.md) are how the crate is
# built, not part of it.
#
# `cargo package --list` reads the manifest and the tree only, so it works on
# a release commit whose version is not on crates.io yet.
#
# Usage: .github/scripts/check-package-contents.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

fail=0
listed="$(cargo package --list --allow-dirty -p lagrangian)"

for must in README.md LICENSE-MIT LICENSE-APACHE Cargo.toml src/lib.rs; do
  if ! printf '%s\n' "$listed" | grep -Fxq -- "$must"; then
    echo "::error::package contents: lagrangian does not ship $must" >&2
    fail=1
  fi
done

stray="$(printf '%s\n' "$listed" | grep -v -E '^(\.cargo_vcs_info\.json|Cargo\.toml|Cargo\.toml\.orig|Cargo\.lock|README\.md|LICENSE-MIT|LICENSE-APACHE|src/.*\.rs|examples/.*\.rs|tests/.*\.rs)$' || true)"
if [ -n "$stray" ]; then
  echo "::error::package contents: lagrangian ships files outside the allowlist; exclude them in its Cargo.toml or widen the list here on purpose:" >&2
  printf '%s\n' "$stray" | sed 's/^/  /' >&2
  fail=1
fi

# A symlink that became a text file holding its target's path still lists
# under the right name. The README must be the README.
if [ ! -s crates/lagrangian/README.md ] || ! cmp -s crates/lagrangian/README.md README.md; then
  echo "::error::package contents: crates/lagrangian/README.md is not the repository README (is the symlink intact?)" >&2
  fail=1
fi
for lic in LICENSE-MIT LICENSE-APACHE; do
  if ! cmp -s "crates/lagrangian/$lic" "$lic"; then
    echo "::error::package contents: crates/lagrangian/$lic is not the repository's $lic (is the symlink intact?)" >&2
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  count="$(printf '%s\n' "$listed" | wc -l | tr -d ' ')"
  echo "package contents: lagrangian ships its README and both licenses, and nothing outside the allowlist ($count files)"
fi
exit "$fail"
