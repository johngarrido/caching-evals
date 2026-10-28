#!/usr/bin/env python3
"""
Harness for measuring Gemini implicit context caching on Vertex AI.

Runs N trials of a matched pair: two requests sharing a byte-identical large
leading prefix but ending with a different trailing task ("cross-tail"), and
reports how often the second one comes back with a prefix cache hit.

  ./run.sh --project my-project                       # both arms, 5 trials each
  ./run.sh --project my-project --reps 10 --settle-ms 30000
  ./run.sh --dry-run                                  # no API calls, synthetic run

Authentication is Application Default Credentials only:
  gcloud auth application-default login

Every run is also written to runs/<run id>.jsonl as it happens, one record per
line, which is what workbench.html reads (live or after the fact).

See README.md for the method.
"""

import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import google.auth
from google import genai
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types

# How each surface issues the matched pair. These are two different mechanisms, not two dialects of
# one. generate_content resends the whole prefix on both requests and relies on implicit prefix
# caching. interactions sends the prefix once and chains the second turn by id, so the server
# holds the conversation; the second turn's cached percentage is measured against a prompt the
# client never sent. Do not read the two methods' hit rates as one number — see README.
SURFACES = {
    'vertex': 'generate_content',
    'vertex_interactions': 'interactions',
}

# Vertex's session-affinity hint: a request body field, sent per request via http_options.extra_body
# (not a header, and not set on the client). Opt-in via --session-affinity; one id for the whole
# run, so every request claims the same session.
#
# This is an allowlisted feature. On a project without it the field is accepted and ignored, and the
# run then looks exactly like a run without it — which is why the id is recorded and printed on the
# workbench plate. A null result here means "no effect observed OR not enabled", never just the first.
#
# generate_content only: interactions.create rejects it with 400 Unknown parameter 'session_id'.
SESSION_FIELD = 'session_id'

SYSTEM = 'You are a careful assistant. Use only the provided context. Answer in short JSON.'
TAIL_A = '### TASK A\nExtract the key facts and list the open issues. JSON: {facts, issues}.'
TAIL_B = '### TASK B\nBased on the SAME context above, produce a plan. JSON: {title, steps}.'

# Output is not what is measured, but it has to finish. A chained turn cannot start from an
# interaction that was cut off (status 'incomplete': "can't start a new turn"), and thinking
# (left at the model's API default) spends this budget before any answer is written.
MAX_OUTPUT_TOKENS = 1024

# Bump when the shape of a run record changes, so the workbench can refuse a file it cannot read.
RUN_SCHEMA = 1

AUTH_HINT = ('Authenticate with Application Default Credentials (the default):\n'
             '  gcloud auth application-default login\n'
             'and pass --project (or set GOOGLE_CLOUD_PROJECT).\n'
             'Or use a Vertex AI API key instead: set VERTEX_API_KEY (or pass --api-key).')

# A Vertex AI API key, read only from this variable or --api-key. GOOGLE_API_KEY / GEMINI_API_KEY are
# deliberately ignored, so an unrelated key in the environment can never silently switch the run off ADC.
API_KEY_ENV = 'VERTEX_API_KEY'

Usage = Dict[str, Optional[float]]


def session_body(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    return {SESSION_FIELD: session_id} if session_id else None


def method_for(surface: str) -> str:
    return SURFACES[surface]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def sdk_version() -> Optional[str]:
    return getattr(genai, '__version__', None)


class RunWriter:
    """Appends one JSON record per line as the run happens, and keeps runs/index.json current.

    Written incrementally and flushed per record so the workbench can follow a run in progress;
    a run that crashes halfway still leaves every trial it completed on disk.
    """

    def __init__(self, out_dir: str, run_id: str) -> None:
        self.run_id = run_id
        self.dir = out_dir
        self.path = os.path.join(out_dir, f'{run_id}.jsonl')
        self.index_path = os.path.join(out_dir, 'index.json')
        os.makedirs(out_dir, exist_ok=True)
        self._fh = open(self.path, 'a', encoding='utf-8')

    def emit(self, record: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, separators=(',', ':')) + '\n')
        self._fh.flush()

    def index(self, entry: Dict[str, Any]) -> None:
        """Upsert this run into the index the workbench reads to populate its run picker."""
        entries: List[Dict[str, Any]] = []
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, encoding='utf-8') as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    entries = [e for e in loaded if isinstance(e, dict) and e.get('run_id') != self.run_id]
            except (json.JSONDecodeError, OSError):
                entries = []  # a corrupt index is rebuilt rather than allowed to abort the run
        entries.append(entry)
        entries.sort(key=lambda e: str(e.get('started_at', '')), reverse=True)
        tmp = self.index_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(entries, fh, indent=2)
        os.replace(tmp, self.index_path)

    def close(self) -> None:
        self._fh.close()


class NullWriter:
    """Stands in for RunWriter under --no-record, so main() never branches on recording."""

    run_id = None
    path = None

    def emit(self, record: Dict[str, Any]) -> None:
        pass

    def index(self, entry: Dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Measure the implicit prefix-cache hit rate of Gemini on Vertex AI.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'surfaces:\n'
            '  vertex                generate_content, full prefix resent on both requests\n'
            '  vertex_interactions   Interactions API, turn 2 chained by previous_interaction_id\n'
            '\n'
            'The two surfaces measure different mechanisms: on the interactions arm the prefix is\n'
            'sent once and the second turn is chained server-side, so its cached percentage is not\n'
            'directly comparable to the stateless cross-tail arm.\n'
            '\n'
            + AUTH_HINT + '\n'
        ),
    )
    parser.add_argument('--project', default=None,
                        help='GCP project id (default: $GOOGLE_CLOUD_PROJECT, then the ADC project)')
    parser.add_argument('--api-key', default=None,
                        help=f'Vertex AI API key, used instead of ADC (default: ${API_KEY_ENV}). With '
                             '--project it authenticates the normal project endpoints; without one it '
                             'runs in express mode (global, vertex arm only). Prefer the env var: the '
                             'flag lands in shell history.')
    parser.add_argument('--model', default='gemini-3.8-flash', help='Model to evaluate (default: gemini-3.8-flash)')
    parser.add_argument('--location', default='global', help='Vertex AI location (default: global)')
    parser.add_argument('--api-version', default='',
                        help='Vertex apiVersion, e.g. v1 or v1beta1 (default: unset, the SDK default)')
    parser.add_argument('--reps', type=int, default=5, help='Number of trials per surface (default: 5)')
    parser.add_argument('--settle-ms', type=int, default=8000, help='Settle time between CASE and PLAN in ms (default: 8000)')
    parser.add_argument('--prefix-lines', type=int, default=1700, help='Number of synthetic prefix lines (~76k tokens) (default: 1700)')
    parser.add_argument(
        '--surfaces',
        default=','.join(SURFACES),
        help=f'Comma-separated surfaces to test (default: {",".join(SURFACES)})',
    )
    parser.add_argument(
        '--order',
        choices=('interleaved', 'blocked'),
        default='interleaved',
        help='interleaved: one trial per arm per round, arm order shuffled each round, so drift during '
             'the run lands evenly on every arm. blocked: all trials of one arm, then the next. '
             '(default: interleaved)',
    )
    parser.add_argument(
        '--session-affinity',
        action='store_true',
        help=f'Send {SESSION_FIELD} in extra_body on every request of the vertex arm, one id for the '
             'whole run. Allowlisted feature: silently ignored on a project that is not enabled.',
    )
    parser.add_argument('--out-dir', default='runs', help='Directory for JSONL run records (default: runs)')
    parser.add_argument('--no-record', action='store_true', help='Print only; write no run record')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Call no API. Emit a synthetic run (marked synthetic everywhere) to exercise the workbench.',
    )
    return parser.parse_args()


def new_nonce() -> str:
    return str(uuid.uuid4())


def user_text(text: str) -> types.Content:
    return types.Content(role='user', parts=[types.Part.from_text(text=text)])


def build_prefix(nonce: str, prefix_lines: int) -> str:
    lines = [
        '### LARGE CACHEABLE PREFIX (byte-identical across the two sibling requests in a trial)',
        f'UNIQUE_NONCE_PER_TRIAL={nonce}',
        '',
    ]
    for i in range(prefix_lines):
        lines.append(
            f'Clause {i}: the parties hereby agree, under section {i % 97}, that obligation number {(i * 7) % 1000} '
            f'shall be performed in good faith, subject to the conditions precedent enumerated in paragraph {(i * 13) % 500}.'
        )
    lines.append('')
    lines.append('### END OF CACHEABLE PREFIX')
    return '\n'.join(lines)


def resolve_project(flag: Optional[str]) -> str:
    """--project, then GOOGLE_CLOUD_PROJECT, then whatever project ADC carries. ADC must exist either way."""
    try:
        _, adc_project = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
    except DefaultCredentialsError as e:
        raise SystemExit(f'No Application Default Credentials found.\n{AUTH_HINT}') from e
    project = flag or os.environ.get('GOOGLE_CLOUD_PROJECT') or adc_project
    if not project:
        raise SystemExit(f'No GCP project set.\n{AUTH_HINT}')
    return project


def make_client(project: Optional[str], location: str, api_version: str,
                api_key: Optional[str]) -> genai.Client:
    http_options = types.HttpOptions(api_version=api_version) if api_version else None
    if api_key and not project:
        # Express mode: the key alone, global publisher endpoint, no project in the path.
        return genai.Client(vertexai=True, api_key=api_key, http_options=http_options)
    # With a project, requests use projects/{p}/locations/{l}/... paths; the key (if any) replaces ADC
    # as the credential. That path is also the only one that serves the Interactions API.
    return genai.Client(vertexai=True, api_key=api_key, project=project, location=location,
                        http_options=http_options)


def to_usage(prompt: Optional[int], cached: Optional[int]) -> Usage:
    # prompt present + cached absent (None) => 0 cached (a genuine miss). Both absent => None (invalid/excluded).
    if prompt is not None and cached is not None:
        pct = round((cached / prompt) * 100, 2)
    elif prompt is not None:
        pct = 0.0
    else:
        pct = None

    return {'prompt': prompt, 'cached': cached, 'pct': pct}


def cached_pct(u: Optional[types.GenerateContentResponseUsageMetadata]) -> Usage:
    return to_usage(
        getattr(u, 'prompt_token_count', None) if u else None,
        getattr(u, 'cached_content_token_count', None) if u else None,
    )


def interaction_usage(interaction: Any) -> Usage:
    """Same shape as cached_pct, read off the Interactions API's own usage block.

    total_input_tokens counts what the server assembled for the turn, which on a chained turn
    includes the prefix the client did not resend. That is what makes this arm worth running.
    """
    u = getattr(interaction, 'usage', None)
    return to_usage(
        getattr(u, 'total_input_tokens', None) if u else None,
        getattr(u, 'total_cached_tokens', None) if u else None,
    )


class Is404Error(Exception):
    pass


class AuthError(Exception):
    """401/403: the credential or its permissions are wrong. Ends the run; retrying cannot fix it."""


class SyntheticClient:
    """Stands in for a real client under --dry-run: no network, no credentials, no spend.

    It fabricates usage so the workbench can be exercised end to end. Every record it produces is
    stamped synthetic, because a made-up measurement that reads as a real one is the one thing this
    project must never put in front of anyone.
    """

    # Arbitrary, and the same for every surface: it exists only to exercise the hit and miss paths.
    # It is not an expectation of what Vertex does.
    HIT_ODDS = 0.5

    def __init__(self, surface: str, prefix_lines: int) -> None:
        self.rng = random.Random(f'{surface}:{prefix_lines}')
        self.prompt_tokens = prefix_lines * 45 + 20

    def usage(self, is_plan: bool) -> Usage:
        prompt = self.prompt_tokens + self.rng.randint(0, 8)
        cached = None
        if is_plan and self.rng.random() < self.HIT_ODDS:
            cached = int(prompt * 0.9639)
        pct = round((cached / prompt) * 100, 2) if cached is not None else 0.0
        return {'prompt': prompt, 'cached': cached, 'pct': pct}


def call_with_retry(fn: Any) -> Tuple[Any, int, float]:
    """Returns (result, retries, start of the attempt that succeeded, monotonic seconds).

    A 404 is raised as Is404Error; a 429 is backed off; the rest raises. The retry count and the
    start time go into the trial record: a backoff on PLAN stretches the CASE->PLAN gap, which is
    the variable --settle-ms controls, so it must never happen invisibly.
    """
    for attempt in range(5):
        started = time.monotonic()
        try:
            return fn(), attempt, started
        except Exception as e:
            # generate_content raises errors.APIError (.code); interactions raises its own client
            # error (.status_code). Match on the HTTP status either way, never on message text.
            code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            if code == 404:
                raise Is404Error(str(e)[:200]) from e
            if code in (401, 403):
                raise AuthError(f'{code}: {str(e)[:300]}') from e
            if (code == 429 or getattr(e, 'status', None) == 'RESOURCE_EXHAUSTED') and attempt < 4:
                time.sleep(10.0 * (attempt + 1))
                continue
            raise

    raise RuntimeError('unreachable')


Call = Tuple[Usage, int, float]  # usage, retries, start of the successful attempt


def generate(client: Any, model: str, contents: List[types.Content], max_out: int, is_plan: bool,
             session_id: Optional[str]) -> Call:
    if isinstance(client, SyntheticClient):
        return client.usage(is_plan), 0, time.monotonic()
    body = session_body(session_id)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        temperature=0.0,
        max_output_tokens=max_out,
        http_options=types.HttpOptions(extra_body=body) if body else None,
    )
    response, retries, started = call_with_retry(lambda: client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    ))
    return cached_pct(response.usage_metadata), retries, started


def interact(client: Any, model: str, max_out: int, is_plan: bool, **kwargs: Any) -> Tuple[Any, Call]:
    """One interactions.create turn. Returns the interaction (for chaining) and the call record."""
    if isinstance(client, SyntheticClient):
        return None, (client.usage(is_plan), 0, time.monotonic())
    turn, retries, started = call_with_retry(lambda: client.interactions.create(
        model=model,
        system_instruction=SYSTEM,
        generation_config={'max_output_tokens': max_out, 'temperature': 0.0},
        store=True,
        **kwargs,
    ))
    return turn, (interaction_usage(turn), retries, started)


Trial = Tuple[Usage, Usage, int, int]  # case usage, plan usage, retries, measured CASE->PLAN gap in ms


def settle_then(settle_s: float, plan: Any) -> Tuple[Call, float]:
    """Sleep the settle, run PLAN, and return its call record plus when the wait began."""
    case_end = time.monotonic()
    time.sleep(settle_s)
    return plan(), case_end


def trial_generate_content(cl: Any, model: str, prefix_text: str, max_out: int, settle_s: float,
                           session_id: Optional[str]) -> Trial:
    """The stateless cross-tail pair: both requests carry the whole prefix, only the tail differs."""
    prefix = user_text(prefix_text)
    case_u, r1, _ = generate(cl, model, [prefix, user_text(TAIL_A)], max_out, False, session_id)
    (plan_u, r2, plan_start), case_end = settle_then(
        settle_s, lambda: generate(cl, model, [prefix, user_text(TAIL_B)], max_out, True, session_id))
    return case_u, plan_u, r1 + r2, round((plan_start - case_end) * 1000)


def trial_interactions(cl: Any, model: str, prefix_text: str, max_out: int, settle_s: float,
                       session_id: Optional[str]) -> Trial:
    """The chained pair: CASE carries the prefix, PLAN carries only the new task plus a turn id.

    store=True (in interact) is load-bearing, not decoration — an interaction that is not stored
    cannot be referenced by previous_interaction_id, so the second turn would have nothing to chain to.
    session_id is not sent: interactions.create rejects it.
    """
    turn1, (case_u, r1, _) = interact(cl, model, max_out, False, input=f'{prefix_text}\n\n{TAIL_A}')
    (_, (plan_u, r2, plan_start)), case_end = settle_then(
        settle_s, lambda: interact(cl, model, max_out, True, input=TAIL_B,
                                   previous_interaction_id=turn1.id if turn1 else None))
    return case_u, plan_u, r1 + r2, round((plan_start - case_end) * 1000)


TRIAL_RUNNER = {'generate_content': trial_generate_content, 'interactions': trial_interactions}


def schedule(surfaces: List[str], reps: int, order: str, rng: random.Random) -> List[Tuple[str, int]]:
    """The (surface, rep) sequence the run executes."""
    if order == 'blocked':
        return [(s, rep) for s in surfaces for rep in range(1, reps + 1)]
    out: List[Tuple[str, int]] = []
    for rep in range(1, reps + 1):
        rnd = surfaces[:]
        rng.shuffle(rnd)
        out.extend((s, rep) for s in rnd)
    return out


def main() -> None:
    args = parse_args()
    model = args.model
    location = args.location
    api_version = args.api_version.strip()
    reps = max(1, args.reps)
    settle_ms = max(0, args.settle_ms)
    prefix_lines = max(1000, args.prefix_lines)
    surfaces = list(dict.fromkeys(s.strip() for s in args.surfaces.split(',') if s.strip()))
    unknown = [s for s in surfaces if s not in SURFACES]
    if unknown or not surfaces:
        raise SystemExit(f'unknown surface(s): {",".join(unknown) or "(none)"}; valid: {",".join(SURFACES)}')
    max_out = MAX_OUTPUT_TOKENS

    api_key = (args.api_key or os.environ.get(API_KEY_ENV) or '').strip() or None
    auth = 'api_key' if api_key else 'adc'
    project: Optional[str]
    if args.dry_run:
        settle_ms = min(settle_ms, 400)  # a dry run is for watching the panel, not for waiting
        project = args.project or 'dry-run'
        print('DRY RUN — no API calls, fabricated usage, every record stamped synthetic\n')
    elif api_key:
        # No ADC lookup: with a key, a project is optional. Without one the run is in express mode.
        project = args.project or os.environ.get('GOOGLE_CLOUD_PROJECT') or None
        if not project and location != 'global':
            raise SystemExit(f'--location {location} needs --project when using an API key '
                             '(express mode, key without a project, is global only).')
        if not project and 'vertex_interactions' in surfaces:
            raise SystemExit('vertex_interactions needs --project when using an API key: express mode '
                             '(key without a project) has no Interactions API route. Pass --project, '
                             'or run --surfaces vertex.')
    else:
        project = resolve_project(args.project)

    # One id for the whole run, so every request on the vertex arm claims the same session.
    session_id = f'cache-eval-{uuid.uuid4().hex[:12]}' if args.session_affinity else None

    started_at = utc_now()
    run_id = f"{started_at.replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:6]}"
    protocol = {
        'model': model,
        'location': location,
        'auth': auth,  # never the key itself
        'api_version': api_version,
        'reps': reps,
        'settle_ms': settle_ms,
        'prefix_lines': prefix_lines,
        'max_output_tokens': max_out,
        'temperature': 0.0,
        'surfaces': surfaces,
        'order': args.order,
        'session_affinity': session_id,
        'methods': {s: method_for(s) for s in surfaces},
        'sdk': 'google-genai',
        'sdk_version': sdk_version(),
        'python': sys.version.split()[0],
        'synthetic': bool(args.dry_run),
    }
    index_entry = {'run_id': run_id, 'file': f'{run_id}.jsonl', 'started_at': started_at,
                   'model': model, 'location': location, 'reps': reps, 'surfaces': surfaces,
                   'synthetic': bool(args.dry_run), 'session_affinity': bool(session_id)}

    writer: Any = NullWriter() if args.no_record else RunWriter(args.out_dir, run_id)
    writer.emit({'type': 'run_start', 'schema': RUN_SCHEMA, 'run_id': run_id, 'started_at': started_at, **protocol})
    writer.index({**index_entry, 'status': 'running'})

    print(f"auth={auth} project={project or '(express mode)'} model={model} location={location} apiVersion={api_version or '(SDK default)'}")
    print(f"reps={reps} settleMs={settle_ms} prefixLines={prefix_lines} order={args.order}")
    print(f"surfaces={','.join(surfaces)}")
    if session_id:
        print(f"sessionAffinity=extra_body.{SESSION_FIELD}={session_id} (vertex arm only;"
              f" allowlisted feature — silently ignored if the project is not enabled)")
    print(f"run={run_id}" + (f" -> {writer.path}\n" if writer.path else " (not recorded)\n"))

    client: Any = None if args.dry_run else make_client(project, location, api_version, api_key)
    arms: Dict[str, Dict[str, Any]] = {}
    for surface in surfaces:
        method = method_for(surface)
        # generate_content only: interactions.create 400s on the field.
        arm_session = session_id if method == 'generate_content' else None
        arms[surface] = {'method': method, 'session': arm_session, 'closed': False,
                         'client': SyntheticClient(surface, prefix_lines) if args.dry_run else client,
                         'hits': 0, 'n': 0, 'pcts': [], 'excluded': 0, 'note': None}
        writer.emit({'type': 'surface_start', 'surface': surface, 'api_version': api_version,
                     'method': method, 'session_affinity': arm_session, 'ts': utc_now()})

    current = None
    for surface, rep in schedule(surfaces, reps, args.order, random.Random(run_id)):
        arm = arms[surface]
        if arm['closed']:
            continue
        if args.order == 'blocked' and surface != current:
            print(f"=== {surface} ===" + ('  [chained]' if arm['method'] == 'interactions' else ''))
            current = surface
        label = f"  {surface} trial {rep}" if args.order == 'interleaved' else f"  trial {rep}"
        kind = 'chained' if arm['method'] == 'interactions' else 'cross-tail'
        prefix_text = build_prefix(new_nonce(), prefix_lines)
        try:
            case_u, plan_u, retries, gap_ms = TRIAL_RUNNER[arm['method']](
                arm['client'], model, prefix_text, max_out, settle_ms / 1000.0, arm['session'])
            timing = {'retries': retries, 'gap_ms': gap_ms}
            timing_str = f" gap={gap_ms}ms" + (f" retries={retries}" if retries else '')

            if plan_u['pct'] is None:  # no usage returned -> invalid, exclude
                arm['excluded'] += 1
                print(f"{label}: EXCLUDED (PLAN returned no usage){timing_str}")
                writer.emit({'type': 'trial', 'surface': surface, 'rep': rep, 'status': 'excluded',
                             'reason': 'PLAN returned no usage', 'case': case_u, 'plan': plan_u,
                             **timing, 'ts': utc_now()})
            else:
                hit = plan_u['pct'] > 50
                arm['n'] += 1
                arm['pcts'].append(plan_u['pct'])
                arm['hits'] += int(hit)
                print(
                    f"{label}: CASE prompt={case_u['prompt']} cached={case_u['cached']} ({case_u['pct']}%) | "
                    f"PLAN({kind}) prompt={plan_u['prompt']} cached={plan_u['cached']} ({plan_u['pct']}%) "
                    f"{'HIT' if hit else 'miss'}{timing_str}"
                )
                writer.emit({'type': 'trial', 'surface': surface, 'rep': rep,
                             'status': 'hit' if hit else 'miss', 'reason': None,
                             'case': case_u, 'plan': plan_u, **timing, 'ts': utc_now()})

            time.sleep(0.2 if args.dry_run else 2.0)
        except Is404Error as e:
            if rep == 1:
                # On the first trial a 404 is a configuration finding, not noise: the model is not
                # served at this location/apiVersion. Close the arm rather than burn its allocation.
                reason = f'404 Not Found: {model} not served at {location}' + \
                         (f' (apiVersion {api_version})' if api_version else '')
                arm['note'], arm['closed'] = reason, True
                print(f"{label}: {reason} — arm closed")
                writer.emit({'type': 'trial', 'surface': surface, 'rep': rep, 'status': 'not_served',
                             'reason': reason, 'case': None, 'plan': None, 'ts': utc_now()})
            else:
                # The arm has already been served, so a later 404 is transient: one excluded trial.
                arm['excluded'] += 1
                print(f"{label}: EXCLUDED (404 after earlier trials succeeded: {str(e)[:300]})")
                writer.emit({'type': 'trial', 'surface': surface, 'rep': rep, 'status': 'excluded',
                             'reason': f'404 mid-run: {str(e)[:300]}', 'case': None, 'plan': None,
                             'ts': utc_now()})
        except AuthError as e:
            writer.emit({'type': 'run_end', 'ended_at': utc_now(), 'ts': utc_now(), 'aborted': str(e)})
            writer.index({**index_entry, 'ended_at': utc_now(), 'status': 'aborted'})
            writer.close()
            raise SystemExit(f'Authentication failed ({auth}) on {surface}: {e}\n\n{AUTH_HINT}')
        except Exception as e:
            arm['excluded'] += 1
            msg = (str(e) or repr(e))[:300]
            print(f"{label}: EXCLUDED (error: {msg})")
            writer.emit({'type': 'trial', 'surface': surface, 'rep': rep, 'status': 'excluded',
                         'reason': msg, 'case': None, 'plan': None, 'ts': utc_now()})

    for surface, a in arms.items():
        writer.emit({'type': 'surface_end', 'surface': surface, 'hits': a['hits'], 'analysed': a['n'],
                     'excluded': a['excluded'], 'allocated': reps, 'pcts': a['pcts'],
                     'method': a['method'], 'note': a['note'], 'ts': utc_now()})

    print('\n=== PREFIX REUSE HIT RATE (PLAN cached% > 50) ===')
    width = max(len(s) for s in arms) + 2
    for s, a in arms.items():
        tag = '  [chained]' if a['method'] == 'interactions' else ''
        if a['note']:
            print(f"  {s.ljust(width)} {a['note']}{tag}")
            continue
        mean = f"{sum(a['pcts']) / a['n']:.1f}%" if a['n'] else 'n/a'
        exc = f"  excluded={a['excluded']}" if a['excluded'] else ''
        pcts_str = ', '.join(str(p) for p in a['pcts'])
        print(f"  {s.ljust(width)} {a['hits']}/{a['n']}  meanCached={mean}  pcts=[{pcts_str}]{exc}{tag}")
    if any(a['method'] == 'interactions' for a in arms.values()):
        print('\n  [chained] arms sent the prefix once and chained turn 2 by id. Their cached %')
        print('  is measured against a prompt the client never resent — read them apart from the rest.')

    ended_at = utc_now()
    writer.emit({'type': 'run_end', 'ended_at': ended_at, 'ts': ended_at})
    writer.index({**index_entry, 'ended_at': ended_at, 'status': 'complete'})
    writer.close()
    if writer.path:
        print(f"\nrecorded -> {writer.path}")


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print('fatal', exc, file=sys.stderr)
        sys.exit(1)
