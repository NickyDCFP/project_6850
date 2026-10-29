"""Run every reproduction script in order: python -m reproduction.run_all"""
import subprocess
import sys

STEPS = [
    ['reproduction.table1_labels'],
    ['reproduction.table2_homophily', '--version', 'legacy'],
    ['reproduction.table2_homophily', '--version', 'legacy', '--corpus', 'full'],
    ['reproduction.table2_homophily', '--version', 'paper-definition'],
    ['reproduction.table7_reachability'],
    ['reproduction.table3_bridges'],
]

for step in STEPS:
    print(f'\n$ python -m {" ".join(step)}', flush=True)
    subprocess.run([sys.executable, '-m', *step], check=True)
