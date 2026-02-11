#!/usr/bin/env python3
import json,os
from collections import defaultdict
backup='data/dblp_conferences_papers_backup.jsonl'
conf_dir='conf_entries'
checks=[('neurips',2022),('neurips',2023),('neurips',2024),('aies',2023),('aies',2024)]+[('fat',y) for y in range(2020,2026)]
conf_keys = {}
for conf,yr in checks:
    path=f'{conf_dir}/{conf}_{yr}.jsonl'
    keys=set()
    if os.path.exists(path):
        with open(path,'r',encoding='utf-8') as fh:
            for line in fh:
                try:
                    o=json.loads(line)
                except Exception:
                    continue
                k=o.get('paper_id') or o.get('key') or o.get('id') or ''
                if k:
                    keys.add(str(k))
    conf_keys[(conf,yr)] = keys

stats=defaultdict(lambda: {'backup_long_abstracts':0,'backup_keys_in_conf':0,'total_backup_entries':0})
with open(backup,'r',encoding='utf-8') as fh:
    for line in fh:
        try:
            obj=json.loads(line)
        except Exception:
            continue
        v=str(obj.get('venue') or obj.get('booktitle') or obj.get('venue_raw') or '').lower()
        url=str(obj.get('url') or obj.get('ee') or '').lower()
        k=str(obj.get('paper_id') or obj.get('key') or obj.get('id') or '')
        year = obj.get('year')
        try:
            year=int(year)
        except Exception:
            year=None
        abstract = (obj.get('abstract') or '')
        for conf,yr in checks:
            tokens=[conf]
            if conf=='neurips': tokens.append('nips')
            if year==yr and (any(t in v for t in tokens) or any(t in k.lower() for t in tokens) or any(t in url for t in tokens)):
                stats[(conf,yr)]['total_backup_entries'] += 1
                if isinstance(abstract,str) and len(abstract)>=40 and len(abstract)!=195:
                    stats[(conf,yr)]['backup_long_abstracts'] += 1
                    if k and k in conf_keys[(conf,yr)]:
                        stats[(conf,yr)]['backup_keys_in_conf'] += 1

for conf,yr in checks:
    s=stats[(conf,yr)]
    print(f"{conf}_{yr}: backup_entries={s['total_backup_entries']}, backup_long_abstracts={s['backup_long_abstracts']}, backup_keys_in_conf={s['backup_keys_in_conf']}")
