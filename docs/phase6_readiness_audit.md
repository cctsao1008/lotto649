# Lotto649 Phase 6.2 — Shadow Forecast Readiness Audit

Status: **implementation committed; Windows verification pending (83 expected tests). GitHub Issue #1 creation time independently verified.**

## Why this gate exists

Forecast #001 was sealed for **2026-11-10** and already pushed in immutable-reference commit
[`d4a131ad6bc503bc7360eb680bab899318a84234`](https://github.com/cctsao1008/lotto649/commit/d4a131ad6bc503bc7360eb680bab899318a84234).

- The sealed source has **2,185 draws**, ending on **2026-11-06** (`115000103`).
- The payload SHA-256 is `99eff658789131708533638014c1e7c646b098b4d473a3099dc1d62ac359c3d6`.
- The research cutoff for the target draw is **2026-11-10 19:00 Asia/Taipei**, not the official drawing time.
- GitHub [Issue #1](https://github.com/cctsao1008/lotto649/issues/1) independently reports a server-created timestamp of **2026-11-06T17:51:07Z** (2026-11-07 01:51:07 Taipei), which precedes the declared research cutoff. Issue contents are editable, so check the edit history or obtain a stronger timestamp attestation when necessary.

A file named `lotto649_recent_20261110.npy` was produced on November 7 using the same two-result API snapshot. User-side read-only comparison showed:

- `lotto649_recent_20261107.npy` and `lotto649_recent_20261110.npy` both have **2,185 records**.
- Their SHA-256 is identical: `2e21fe66094ef256dc76fc77d5c324195071bc70d3642b68a675d35ee9e480c1`.
- Both end with the **2026-11-06** result.
- The `after_20261110.json` API snapshot contained only November 3 and 6 records, not the November 10 result.

The `lotto_shadow.py score` refusal before the research deadline was correct and expected. However, the filename confusion merits its own **read-only diagnostic**, with no attempt to bypass the timestamp guard or rewrite an already committed forecast.

## Tool: `lotto_shadow_status.py`

`inspect_status(...)` and the corresponding CLI verify:

1. The forecast JSON's canonical payload SHA-256.
2. Optionally, exact byte identity with the ORIGINAL forecast Git commit.
3. Candidate structured NPY validity and the full sealed historical **date/ID/main-six prefix**.
4. Candidate byte SHA-256, draw count, actual latest observed date/ID, and the **first newly appended draw**, if any.
5. Manifest SHA-256 and counts, plus the bytes, hash and parsed record count of every referenced raw API snapshot. Each staged candidate draw must match the raw API record for date, period, main six and special number; mismatches are fatal.
6. The operator-declared research deadline against the machine's current UTC time.

No files are created, deleted, rewritten or renamed, and the command never scores the target draw.

Possible statuses:

- `NO_CANDIDATE_SOURCE`: no candidate file exists.
- `WAITING_FOR_TARGET_DRAW`: candidate has **no observations after** the sealed archive; a misleading name does not change that.
- `BLOCKED_TARGET_NOT_FIRST_NEW_DRAW`: next added draw is for a different date; a one-step forecast must target the first new observation.
- `BLOCKED_RESEARCH_DEADLINE_NOT_PASSED`: target draw appears, but the operator-declared deadline has not passed.
- `READY_FOR_SCORING_PENDING_OFFICIAL_RESULT_CHECK`: candidate has the correct first new date and passes the declared cutoff. **The operator must still verify that the numbers were actually published by the official lottery** before running score.

Invalid source prefixes, bad hashes or invalid API manifest evidence raise errors rather than yielding a favorable readiness status.

## Windows commands (safe to run before the target draw)

```powershell
cd D:\my-github\lotto649
git pull origin main

# Expected: 83 tests (73 previously passed + 10 new readiness tests).
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Read-only audit; may be run at any time.
.\.venv\Scripts\python.exe .\lotto_shadow_status.py `
    --seal data/shadow/seals/forecast_20261110.json `
    --seal-commit d4a131ad6bc503bc7360eb680bab899318a84234 `
    --candidate data/shadow/sources/lotto649_recent_20261110.npy `
    --manifest data/shadow/sources/lotto649_recent_20261110.manifest.json
```

With the November 7 duplicate source, the expected status is `WAITING_FOR_TARGET_DRAW`; `record_count=2185`, `new_draw_count=0`, `last_recorded_date=20261106`, and `same_file_sha256_as_sealed_source=True`. This is a prediction based on the user's earlier comparison, not a claim to have run the new Windows script.

After the actual target draw has been officially published and a fresh API snapshot has been captured, **create a new filename for the fresh source and manifest**. Validate these with the readiness tool first. Only run `lotto_shadow.py score` after checking the official results and the recorded research cutoff; the score command will independently revalidate the sealed prefix and Git bytes.

## Research boundaries

- A server-created GitHub issue can corroborate publication timing, but its body is editable.
- Neither a locally reported time nor Git commit metadata alone proves ex-ante seal publication.
- A zero positive-result count is still a valid outcome; do not alter the probability model after seeing results.
- IID remains the default; Static/Lag/MLP stay diagnostic only.
- All newly authored Python comments, docstrings, and terminal messages are English-only.
- `lotto_select.py` is still deferred.
