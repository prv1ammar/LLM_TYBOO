import json
import re

with open('taxonomy.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append("TAG_SIGNALS_GEN = [")
tags_seen = set()
for item in data:
    tag = str(item.get('Tag', '')).strip()
    if not tag or tag in tags_seen: continue
    tags_seen.add(tag)
    
    val = tag.split(':')[-1].strip().lower()
    val = re.sub(r'[^a-z0-9]', '.*', val)
    if val:
        regex_str = f"r'{val}'"
    else:
        regex_str = "r'()'"
    
    lines.append(f"    ('{tag}', re.compile({regex_str}, re.I)),")

lines.append("]")

with open('generated_tags.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
