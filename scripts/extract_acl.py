import glob
import json
import os
import os.path as osp

def load_jsonl(path):
    """Load a .jsonl file and return a list of JSON objects."""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    """Save a list of JSON objects to a .jsonl file."""
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def reconstruct_from_detailed():
    detailed_logs = glob.glob("data/final_filtered/logs/*_filter_detailed.jsonl")
    for log_file in detailed_logs:
        print(f"Processing {log_file}...")
        conf = osp.basename(log_file).split('_')[0]
        records = load_jsonl(log_file)
        passed = [r for r in records if r.get('filter_result') == 'pass']
        options = {}
        for r in records:
            if 'filter_result' in r:
                options[r['filter_result']] = options.get(r['filter_result'], 0) + 1
            else:
                options['no_result'] = options.get('no_result', 0) + 1
        print("Filter results breakdown:", options)
        save_jsonl(passed, f"data/final_filtered/{conf}_filtered.jsonl")

def pull_from_acl():
    acl_filtered = 'data/filtered_ethics_safety_papers.jsonl'
    acl_plus = load_jsonl(acl_filtered)
    ex_keys = acl_plus[0].keys()
    to_add = []
    for entry in acl_plus:
        if set(entry.keys()) != set(ex_keys):
            continue
        item = {
            'id': entry.get('paper_id'),
            'title': entry.get('title'),
            'authors': entry.get('authors'),
            'abstract': entry.get('abstract'),
        }
        item.update(entry)
        print(item.keys())
        to_add.append(item)
    save_jsonl(to_add, 'data/final_filtered/acl_anthology_filtered.jsonl')
if __name__ == '__main__':

    for item in glob.glob("data/final_filtered/*_filtered.jsonl"):
        print(len(load_jsonl(item)), item)