Requires Python 3.13+, Node 18+, and `OPENAI_API_KEY` set in your environment.

```sh
cd personal_health_agent
python -m pip install -r requirements.txt
git clone https://github.com/yahskapar/personal-health-insights-agent phia
git -C phia checkout c0abed8096641bd6dc2f8154f25558aa76a28453
python experiment.py --bench phia --out results --workers 4 --budget 40
```

This runs control, no review, and decision review on all 4,000 PHIA questions using GPT-6 Luna. Results, accuracy, estimated API cost, and timing are saved in `results`. Use a new output directory each time. Select individual conditions with `--arms control`, `--arms none`, or `--arms decision`.
