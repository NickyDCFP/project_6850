"""Run every analysis script in order: python -m analysis.run_all"""
import subprocess
import sys

STEPS = [
    ['analysis.homophily_over_time', '--version', 'paper-definition'],
    ['analysis.homophily_over_time', '--version', 'paper-definition', '--windows', 'yearly'],
    ['analysis.homophily_over_time', '--version', 'legacy'],
    ['analysis.homophily_over_time', '--version', 'legacy', '--corpus', 'full'],
]

for step in STEPS:
    print(f'\n$ python -m {" ".join(step)}', flush=True)
    subprocess.run([sys.executable, '-m', *step], check=True)
