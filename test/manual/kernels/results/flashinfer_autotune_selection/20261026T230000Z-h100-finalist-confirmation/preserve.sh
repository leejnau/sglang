#!/usr/bin/env bash
# Preserve one completed or failed phase without replacing an earlier archive.
set -euo pipefail
study=$(realpath "${1:?study required}")
attempt=${2:?attempt required}
phase=${3:?admission or timing phase required}
durable=${4:?durable archive directory required}
[[ "$attempt" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 1
[[ "$phase" == admission || "$phase" == timing ]] || exit 1
results="$study/results/finalist-confirmation/$attempt"
stem="20261026T230000Z-h100-finalist-confirmation-$attempt-$phase"
archive="$durable/$stem-full-raw.tgz"
record="$durable/$stem-preservation.json"
[[ -d "$results" && -d "$durable" && ! -e "$archive" && ! -e "$record" ]] || exit 1
python3 - "$results" "$phase" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); phase=sys.argv[2]
target=root/f'preservation-manifest-{phase}.json'
assert not target.exists()
rows=[]
for path in sorted(root.rglob('*')):
    if path.is_file():
        rows.append({'path':str(path.relative_to(root)), 'size':path.stat().st_size,
                     'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
target.write_text(json.dumps({'phase':phase,'files':rows},indent=2)+'\n')
PY
tar -czf "$archive" -C "$study/results/finalist-confirmation" "$attempt"
python3 - "$archive" "$record" "$phase" <<'PY'
import hashlib,json,sys
from pathlib import Path
archive,record=map(Path,sys.argv[1:3])
with record.open('x') as handle:
    json.dump({'phase':sys.argv[3], 'archive':str(archive), 'size':archive.stat().st_size,
               'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},handle,indent=2)
    handle.write('\n')
print(record.read_text())
PY
