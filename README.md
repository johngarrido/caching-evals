# caching-evals

A small harness for measuring **implicit context caching** on **Vertex AI**: how often Gemini
actually reuses a large shared leading prefix, and whether a given setting (model, location,
apiVersion, timing, session affinity) changes that. Run it against your own project and read the
result in the bundled workbench.

The workload it models: a large document sent first, then step 1 "case intake" and step 2 "plan".
Both steps carry the **same document prefix with a different trailing task**. We call that pair
*cross-tail*. If implicit caching picks up the prefix, step 2 is billed far fewer full-price input
tokens.

Python 3.11+, `uv`, and the official `google-genai` SDK (pinned in `uv.lock`). The prefix is
synthetic. No real data is sent.

---

## Prerequisites

- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)
- The [`gcloud` CLI](https://cloud.google.com/sdk/docs/install)
- A GCP project with the **Vertex AI API** enabled, and an identity holding
  `roles/aiplatform.user` (or broader) on it

## Authentication

### Default: Application Default Credentials (ADC)

By default the harness authenticates through Application Default Credentials. It reads no
service-account JSON and no `.env` file. Log in once:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

The project is taken from `--project`, then `GOOGLE_CLOUD_PROJECT`, then the project stored with
your ADC. If ADC or a project is missing, the run stops before sending any request and tells you
what to fix.

### Optional: Vertex AI API key

If you have a [Vertex AI API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys),
set it in `VERTEX_API_KEY` and it is used **instead of** ADC:

```bash
export VERTEX_API_KEY=...        # prefer this over --api-key, which lands in shell history

./run.sh --surfaces vertex --session-affinity --reps 10 --project YOUR_PROJECT_ID
```

- **With `--project`** (or `GOOGLE_CLOUD_PROJECT`), requests go to the normal project endpoints,
  authenticated by the key. Both arms and any `--location` work.
- **Without a project** the run uses *express mode*: global endpoint, `vertex` arm only. Express
  mode has no Interactions API route, so asking for `vertex_interactions` there stops the run with
  an explanation.

Only `VERTEX_API_KEY` / `--api-key` is read. `GOOGLE_API_KEY` and `GEMINI_API_KEY` are ignored, so a
Gemini API key already in your environment can't silently switch a run off ADC. The run record
stores which auth mode was used (`adc` or `api_key`), never the key. A 401/403 stops the run at
the first request instead of excluding every trial.

## Running it

```bash
uv sync
```

The two default commands, one per arm, 10 trials each:

```bash
# generate_content only (stateless cross-tail), with session affinity
./run.sh --surfaces vertex --session-affinity --reps 10 --project YOUR_PROJECT_ID

# Interactions API only (chained turns)
./run.sh --surfaces vertex_interactions --reps 10 --project YOUR_PROJECT_ID
```

Other variations:

```bash
./run.sh --project YOUR_PROJECT_ID                 # both arms, 5 trials each
./run.sh --project YOUR_PROJECT_ID --reps 10
./run.sh --project YOUR_PROJECT_ID --settle-ms 30000      # widen the gap between the two requests
./run.sh --project YOUR_PROJECT_ID --model gemini-3.5-flash --location us-central1
./run.sh --project YOUR_PROJECT_ID --api-version v1       # pin the Vertex apiVersion
./run.sh --project YOUR_PROJECT_ID --surfaces vertex      # one arm only
./run.sh --dry-run                                 # no API calls, a synthetic run for the workbench
./run.sh --help                                    # every flag
```

| Flag | Default | Meaning |
|---|---|---|
| `--project` | `$GOOGLE_CLOUD_PROJECT` / ADC | GCP project id |
| `--api-key` | `$VERTEX_API_KEY` | Vertex AI API key, used instead of ADC (see above) |
| `--model` | `gemini-3.8-flash` | Model under test |
| `--location` | `global` | Vertex AI location |
| `--api-version` | unset (SDK default) | Vertex apiVersion, e.g. `v1`, `v1beta1` |
| `--surfaces` | `vertex,vertex_interactions` | Arms to run (see below) |
| `--reps` | `5` | Trials per arm |
| `--settle-ms` | `8000` | Wait between the two requests of a trial |
| `--prefix-lines` | `1700` | Size of the synthetic prefix (~76k tokens) |
| `--order` | `interleaved` | `interleaved` or `blocked`; see [Method](#method) |
| `--session-affinity` | off | Send a fixed `session_id` on the `vertex` arm; see below |
| `--out-dir` / `--no-record` | `runs` | Where run records go, or write none |

**Cost.** Every trial sends roughly 2 × 76k input tokens, which are billed. Use `--reps` and
`--prefix-lines` to size a run before scaling it up.

### The two arms

| Arm | API | What it measures |
|---|---|---|
| `vertex` | `models.generate_content` | Stateless cross-tail. Both requests carry the whole prefix, and only the tail differs. This is pure implicit prefix caching. |
| `vertex_interactions` | `interactions.create` | Chained turns. CASE creates a stored interaction carrying the prefix, and PLAN sends only the new task plus `previous_interaction_id`. The server reassembles the conversation. |

```python
turn1 = client.interactions.create(model=..., input=PREFIX + TAIL_A, store=True)
turn2 = client.interactions.create(model=..., input=TAIL_B, previous_interaction_id=turn1.id)
```

These are **different mechanisms**. The chained arm's cached percentage is measured against a
prompt the client never resent, so **do not pool or directly compare the two rates**. The console
tags the chained arm `[chained]`, and the workbench badges it and warns when a run mixes both.
`store=True` is required, because an interaction that is not stored cannot be chained to.

### Session affinity

`--session-affinity` sends `session_id` in the request body of every `vertex` request, with one id
generated per run:

```python
config=types.GenerateContentConfig(
    http_options=types.HttpOptions(extra_body={'session_id': session_id}),
)
```

The hypothesis is that a stable session id steers sibling requests to the same serving replica and
so makes prefix reuse more likely. Trials still each get a unique nonce, so they stay independent.

**It is an allowlisted feature.** On a project that is not enabled, the field is accepted and
silently ignored. A flat result therefore means *no effect **or** not enabled*. Confirm the
allowlist before reading anything into it. It is not sent on `vertex_interactions`, because
`interactions.create` rejects the field (`400 Unknown parameter`). The id is stored in the run
record and shown on the workbench, so a run with affinity on can't be mistaken for one without.

## The workbench

`workbench.html` is a single self-contained file: no build step, nothing to install. It reads the
run records and shows each run as a trial report. You get one row per arm, every trial's outcome,
the 95% Wilson interval on each arm's hit rate, and a ledger of trials that left the denominator
and why.

```bash
python3 -m http.server 8787        # from the repo root
# then open http://127.0.0.1:8787/workbench.html
```

Served this way, it reads `runs/index.json` to fill the run picker. Tick **Follow live** and it
re-reads the current run every few seconds while the harness writes. Opening the file directly
(`file://`) also works. Use **Load .jsonl** to pick a record by hand.

Expanding an arm shows every trial's token counts plus the **measured** CASE→PLAN gap and retry
count. A gap noticeably longer than the configured settle (e.g. after a 429 backoff) is
highlighted.

It deliberately will not show a point estimate where the sample only supports an interval. It
holds no credentials and cannot start a run.

## Method

Each **trial** is independent: a **unique nonce goes at the very top** of a large (~76k token)
synthetic prefix. Implicit caching keys off a shared *leading* prefix, so a unique first line makes
every trial start cold, and trials never contaminate each other.

1. `CASE = [PREFIX, tail_A]` is the cold writer (expect `cached = 0`).
2. Wait `--settle-ms`.
3. `PLAN = [PREFIX, tail_B]` sends the same prefix bytes with a different tail. A hit means
   `cachedContentTokenCount ≈ prefix size`.

- **Hit**: PLAN reports more than 50% of its prompt served from cache. A miss on Vertex returns
  `cachedContentTokenCount` absent, which counts as 0%.
- **Excluded**: the trial errored or returned no usage. It is listed with its reason and left out of
  the denominator, never silently dropped.
- **Not served**: a 404 on an arm's first trial means the model isn't available at that
  location/apiVersion. The arm is closed rather than spending its allocation. A 404 later in a run
  counts as one excluded trial.
- **Retries and gap**: a 429 is backed off and retried. The retry count and the measured CASE→PLAN
  gap are recorded per trial, so a backoff that stretched the gap is always visible.
- **Order**: by default arms are **interleaved**. Each round runs one trial of every arm, in a
  shuffled order, so drift in serving conditions during the run lands evenly on all arms.
  `--order blocked` runs each arm's trials back to back instead.
- **Held constant**: a fixed `system_instruction`, `temperature=0`, thinking left at the model's
  API default, `max_output_tokens=1024` (room for the short answer to finish, since a cut-off
  interaction cannot be chained), and **no `cached_content`**. Only implicit caching is measured, never explicit.

## Reading a result

- Read the **interval**, not just the hit count. With 5–10 trials per arm the interval is wide.
  The workbench only calls an arm reliable or unreliable once its interval clears the 50% line.
- Implicit caching is **best-effort and serving-side**. It can move with the model version,
  region, or time without any change on your side. Record the model, location, SDK version and date
  with any result you keep. The run record and workbench plate capture all of these.
- Compare `vertex` against `vertex` and chained against chained.

## Layout

```
cache_eval.py      the harness
run.sh             uv sync + run
workbench.html     the report viewer: single file, no build, reads runs/
runs/              created on first run (git-ignored)
  index.json       run list, for the workbench's run picker
  *.jsonl          one record per run, one JSON line per event
```

## Access notice

Access to this repository is temporary (30 days). Clone it before **2026-11-26**.

## License

Apache 2.0. See [LICENSE](LICENSE).
