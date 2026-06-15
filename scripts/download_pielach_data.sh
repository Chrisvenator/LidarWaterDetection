#!/usr/bin/env bash
#
# Download the Mandlburger et al. Pielach River reference dataset (Oct 2024)
# from TU Wien Research Data into data/mandlburger_pielach_2024/.
#
#   Record: https://researchdata.tuwien.ac.at/records/taz19-r6618
#   DOI:    10.48436/taz19-r6618   (CC-BY 4.0)
#
# Used to validate this repo's water-vs-land classifier against an
# independent, same-epoch survey (same river, same RIEGL topo-bathy sensor).
#
# Usage:
#   scripts/download_pielach_data.sh [--no-images] [--extract] [--dry-run]
#
#   --no-images   Skip files 09/10/11 (~49 GB of JPGs); grab the ~5 GB of
#                 LiDAR + reference data only. Enough for validation.
#   --extract     7z-extract each archive after download (into same dir).
#   --dry-run     Print what would be downloaded, do nothing.
#
# Resumable: re-run anytime. Completed files (matching expected size) are
# skipped; partial files resume via `curl -C -`.

set -euo pipefail

readonly RECORD="taz19-r6618"
readonly BASE="https://researchdata.tuwien.ac.at/api/records/${RECORD}/files"
readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DEST="${ROOT}/data/mandlburger_pielach_2024"

# filename:expected_size_bytes  (size used to detect a complete download)
readonly FILES=(
  "readme.txt:5421"
  "03_LiDAR_topobathy_classified.7z:193084930"
  "04_LiDAR_topo_classified.7z:201225145"
  "05_water_surface_model.7z:55763"
  "06_control_points.7z:1426"
  "07_underwater_reference_targets.7z:636"
  "08_underwater_transect_points.7z:1933"
  "01_LiDAR_topobathy_raw.7z:2821467142"
  "02_LiDAR_topo_raw.7z:1710997769"
  "09_images_nadir_oblique.7z:25586266119"
  "10_image_sequences_nadir.7z:15331828770"
  "11_synch_images_oblique.7z:8086330156"
)

# Files skipped by --no-images (the large JPG archives).
is_image_file() { [[ "$1" == 09_* || "$1" == 10_* || "$1" == 11_* ]]; }

SKIP_IMAGES=0
DO_EXTRACT=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --no-images) SKIP_IMAGES=1 ;;
    --extract)   DO_EXTRACT=1 ;;
    --dry-run)   DRY_RUN=1 ;;
    -h|--help)   grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -28; exit 0 ;;
    *) echo "Unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

command -v curl >/dev/null || { echo "curl not found" >&2; exit 1; }
mkdir -p "$DEST"

actual_size() { stat -c '%s' "$1" 2>/dev/null || echo 0; }

download_one() {
  local name="$1" want="$2" path="${DEST}/$1"
  if [[ "$(actual_size "$path")" == "$want" ]]; then
    echo "✓ $name (already complete)"
    return 0
  fi
  if (( DRY_RUN )); then
    echo "→ would download $name ($want bytes)"
    return 0
  fi
  echo "↓ $name ..."
  curl -fL -C - --retry 5 --retry-delay 5 "${BASE}/${name}/content" -o "$path"
  local got; got="$(actual_size "$path")"
  if [[ "$got" != "$want" ]]; then
    echo "  ! size mismatch for $name: got $got, expected $want" >&2
    return 1
  fi
}

extract_one() {
  local name="$1" path="${DEST}/$1"
  [[ "$name" == *.7z ]] || return 0
  command -v 7z >/dev/null || { echo "  ! 7z not found, skip extract" >&2; return 0; }
  echo "  unpacking $name"
  7z x -y -o"$DEST" "$path" >/dev/null
}

echo "Destination: $DEST"
for entry in "${FILES[@]}"; do
  name="${entry%%:*}"
  size="${entry##*:}"
  if (( SKIP_IMAGES )) && is_image_file "$name"; then
    echo "skip $name (--no-images)"
    continue
  fi
  download_one "$name" "$size"
  (( DO_EXTRACT )) && extract_one "$name"
done

echo "Done. Files in $DEST"
