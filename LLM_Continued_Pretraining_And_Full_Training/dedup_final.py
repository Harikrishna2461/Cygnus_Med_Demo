import json
from pathlib import Path
import hashlib

def remove_duplicates(input_file, output_file):
    seen = set()
    unique_chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            text_hash = hashlib.sha256(chunk.get('text', '').encode()).hexdigest()
            
            if text_hash not in seen:
                seen.add(text_hash)
                unique_chunks.append(chunk)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in unique_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    return len(unique_chunks)

train_file = Path("data/final_augmented/train.jsonl")
output_file = Path("data/final_augmented/train_dedup.jsonl")

with open(train_file, 'r', encoding='utf-8') as f:
    original_count = sum(1 for _ in f)

final_count = remove_duplicates(train_file, output_file)

print(f"Original chunks:  {original_count}")
print(f"Final chunks:     {final_count}")
print(f"Duplicates:       {original_count - final_count}")

# Replace original with dedup
import os
os.remove(train_file)
os.rename(output_file, train_file)
print(f"Saved to {train_file}")
