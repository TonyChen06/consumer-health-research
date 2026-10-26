import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.request
import pandas as pd
from experiment_review import install
from experiment_score import score

SOURCE = Path(__file__).resolve().parent
MODEL = 'gpt-6-luna'
BENCH = None


def load_profile(profile):
    base = BENCH / 'synthetic_wearable_users'
    summary = pd.read_csv(base / f'summary_df_{profile}.csv')
    activities = pd.read_csv(base / f'exercise_df_{profile}.csv')
    person = summary.iloc[0]
    profile_df = pd.DataFrame([{
        'age': int(person.age), 'gender': person.gender,
        'weight_kg': float(person.weight_kg), 'height_cm': float(person.height_cm),
        'averageDailySteps': float(summary.steps.mean()), 'elderly': bool(person.age > 65),
    }])
    summary = summary.drop(columns=['pseudo_id2','age','gender','weight_kg','height_cm'])
    mapping = {'startTime':'start_time', 'endTime':'end_time', 'activityName':'activity_name',
               'activeZoneMinutes':'active_zone_minutes', 'averageHeartRate':'average_heart_rate',
               'elevationGain':'elevation_gain'}
    activities = activities.rename(columns=mapping).drop(columns=['pseudo_id2','date'])
    for df, cols in [(summary,['datetime','bed_time','wake_up_time']),
                     (activities,['start_time','end_time'])]:
        for col in cols:
            df[col] = pd.to_datetime(df[col]).astype('datetime64[ns]')
    summary.index = summary.datetime
    activities.index = activities.start_time
    population = pd.DataFrame(columns=['percentile','age','gender'])
    return {'summary':summary,'activities':activities,'profile':profile_df,'population':population}


def build_agent(data):
    from pha.agents import DataScienceAgent, DomainExpertAgent, HealthCoachAgent, MultiAgentOrchestrator, is_react_available
    if not is_react_available():
        raise RuntimeError('Full domain expert unavailable; refusing simplified fallback')
    config = dict(api_key=os.environ['OPENAI_API_KEY'], provider='openai', model_name=MODEL)
    ds = DataScienceAgent()
    ds.configure(**config)
    ds.load_dataframes(data)
    de = DomainExpertAgent(search_backend='duckduckgo')
    de.get_agent(**config, exemplar_files=sorted(str(p) for p in (SOURCE/'few_shots').glob('*.ipynb')))
    hc = HealthCoachAgent(simple_mode=True)
    hc.configure(**config)
    orch = MultiAgentOrchestrator(debug_verbose=True)
    orch.configure(**config)
    orch.set_agents(data_science_agent=ds,domain_expert_agent=de,health_coach_agent=hc)
    return orch


TASK_CONTEXT = '''Dataset context: these are historical wearable records from October 2023.
The records are synthetic; fractional counts are valid stored values for this benchmark.
Use 2023-10-31 as the reference date for relative dates. There is no population comparison dataset.
Activity names retain their recorded PHIA labels (for example Run, Walk, Aerobic Workout).
Activities include average_heart_rate (beats/minute), elevation_gain (meters), speed (miles/hour),
distance (miles), duration (minutes), and active_zone_minutes (minutes).
Summary sleep *_percent columns are percentages on a 0-100 scale.
Answer the question using the records. End your final response with exactly one <answer>VALUE</answer>
tag containing only the requested number or activity name(s), without units or explanation.
Round numerical answers to two decimal places, preserving fractional counts.
Use a JSON list for multiple activity names, and null when the requested statistic is undefined.
Question: '''


def manifest():
    names = {'active_achiver': 41, 'inactive_insomniacs': 171,
             'sedentary_sleeper': 333, 'health_behavior': 465}
    rows = []
    for name, profile in names.items():
        with (BENCH / 'data' / 'auto_eval' / f'{name}_final_v2.csv').open(encoding='utf-8-sig') as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                rows.append({'id': f'{profile}:{index:04d}', 'profile': profile,
                             'question': row['question'], 'label': row['label']})
    rows.sort(key=lambda row: hashlib.sha256(('PHA-control-v1|' + row['id']).encode()).hexdigest())
    if len(rows) != 4000 or len({row['id'] for row in rows}) != 4000:
        raise ValueError('Expected the complete 4000-question PHIA benchmark')
    return rows


def worker():
    row = json.load(sys.stdin)
    started = time.time()
    result = {'id': row['id'], 'model': MODEL}
    agent = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            data = load_profile(row['profile'])
            agent = install(build_agent(data), data, mode=row['arm'], threshold=.8)
            result.update(response=agent.respond(TASK_CONTEXT + row['question']),
                          insights=agent.get_agent_insights(), status='completed')
    except Exception as error:
        result.update(status='error', error=type(error).__name__)
    events = getattr(agent.data_science_agent, 'decision_events', []) if agent else []
    if any(event['status'] == 'error' for event in events):
        result.update(status='error', error='Decision failure')
    result.update(elapsed_seconds=time.time() - started, decision_events=events)
    print(json.dumps(result, ensure_ascii=False))


def run(args):
    rows = manifest()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = [SOURCE / name for name in ['experiment.py', 'experiment_review.py',
               'experiment_score.py', 'experiment_transport.mjs', 'requirements.txt']]
    inputs = list((BENCH / 'data' / 'auto_eval').glob('*_final_v2.csv'))
    inputs += [BENCH / 'synthetic_wearable_users' / f'{kind}_df_{profile}.csv'
               for profile in [41, 171, 333, 465] for kind in ['summary', 'exercise']]
    (args.out / 'config.json').write_text(json.dumps({
        'arms': args.arms, 'workers': args.workers, 'model': MODEL, 'threshold': .8,
        'input_sha256': {str(p.relative_to(BENCH)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, indent=2), encoding='utf-8')
    (args.out / 'manifest.json').write_text(json.dumps(rows), encoding='utf-8')
    log = args.out / 'usage.jsonl'
    log.touch()
    host = subprocess.Popen(['node', str(SOURCE / 'experiment_transport.mjs'), str(log), str(args.budget)],
                            stdout=subprocess.PIPE, text=True)
    totals = {}
    try:
        base = host.stdout.readline().strip()
        if not base.startswith('http://127.0.0.1:'):
            raise RuntimeError('API host failed to start')
        env = {k: v for k, v in os.environ.items() if k not in ('OPENAI_API_KEY', 'OPENAI_BASE_URL')}
        env.update(OPENAI_BASE_URL=base, PHA_DECISIONS_BASE_URL=base, MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')

        def case(row, arm):
            cid = arm + '-' + row['id'].replace(':', '_')
            with (args.out / (cid + '.log')).open('w', encoding='utf-8') as handle:
                process = subprocess.Popen([sys.executable, str(SOURCE / 'experiment.py'), '--worker', '--bench', str(BENCH)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=handle, text=True, encoding='utf-8',
                    env={**env, 'OPENAI_API_KEY': 'local-proxy-' + cid}, start_new_session=os.name != 'nt')
                try:
                    output, _ = process.communicate(json.dumps({**row, 'arm': arm}), timeout=600)
                    result = json.loads(output)
                except Exception as error:
                    if process.poll() is None:
                        if os.name == 'nt':
                            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                        else:
                            import signal
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    result = {'status': 'error', 'error': type(error).__name__, 'elapsed_seconds': None}
            result.update(**row, arm=arm)
            result.update(score(result.get('response', ''), row['label']))
            if result['status'] == 'error':
                result['published_label_match'] = False
            (args.out / (cid + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
            with urllib.request.urlopen(base.removesuffix('/v1') + '/status', timeout=5) as response:
                fatal = json.load(response)['fatal']
            return result, fatal

        for arm in args.arms:
            completed = []
            stopped = False
            iterator = iter(rows)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(case, next(iterator), arm) for _ in range(min(args.workers, len(rows)))}
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        result, fatal = future.result()
                        completed.append(result)
                        stopped |= bool(fatal) or result['status'] == 'error'
                    if not stopped:
                        for _ in done:
                            row = next(iterator, None)
                            if row is not None:
                                pending.add(pool.submit(case, row, arm))
                    print(f'{arm}: {len(completed)}/{len(rows)}', flush=True)
            usage = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
            seconds = [r['elapsed_seconds'] for r in completed if r['elapsed_seconds'] is not None]
            totals[arm] = {
                'requested': len(rows), 'completed': len(completed), 'complete': len(completed) == len(rows),
                'accuracy': sum(r['published_label_match'] for r in completed) / len(completed),
                'errors': sum(r['status'] == 'error' for r in completed),
                'format_failures': sum(not r['format_valid'] for r in completed),
                'estimated_usd': sum(r.get('cost', 0) for r in usage if r['case_id'].startswith(arm + '-')),
                'mean_seconds': statistics.mean(seconds) if seconds else None,
                'median_seconds': statistics.median(seconds) if seconds else None,
            }
            (args.out / 'summary.json').write_text(json.dumps(totals, indent=2), encoding='utf-8')
            if stopped:
                raise RuntimeError('Run stopped after an API or worker failure; completed results preserved')
        print(json.dumps(totals, indent=2))
    finally:
        host.terminate()
        host.wait()


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--arms', nargs='+', choices=['control', 'none', 'decision'], default=['control', 'none', 'decision'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--budget', type=float, default=40)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    BENCH = args.bench.resolve()
    args.out = args.out.resolve()
    if not 1 <= args.workers <= 24 or len(set(args.arms)) != len(args.arms):
        parser.error('Use 1-24 workers and distinct arms')
    if args.worker:
        worker()
    else:
        if not os.environ.get('OPENAI_API_KEY'):
            parser.error('Set OPENAI_API_KEY')
        run(args)
