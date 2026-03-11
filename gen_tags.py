import json
import re

with open('taxonomy.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

lines = []
lines.append("import re")
lines.append("TAG_SIGNALS_GEN = [")
tags_seen = set()
for item in data:
    tag = str(item.get('Tag', '')).strip()
    if not tag or tag in tags_seen: continue
    tags_seen.add(tag)
    
    val = tag.split(':')[-1].strip().lower()
    # clean up value
    val = re.sub(r'\s+', ' ', val)
    if not val or val == '-' or len(val) < 2: 
        continue 
        
    # Use word boundaries to avoid matching substrings
    val_escaped = re.escape(val)
    # Special handling for common small strings to ensure they are whole words
    regex_str = f"r'\\b{val_escaped}\\b'"
    
    lines.append(f"    ('{tag}', re.compile({regex_str}, re.I)),")

lines.append("]")

with open('generated_tags.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
