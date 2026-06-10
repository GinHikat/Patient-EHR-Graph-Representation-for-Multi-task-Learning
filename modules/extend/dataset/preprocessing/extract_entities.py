import json
import csv
from collections import defaultdict
import os

input_file = r"data\viettel\vietnamese_ner\unified_qwen_dataset.jsonl"
output_file = r"data\viettel\vietnamese_ner\aggregated_entities.csv"

def extract_unique_entities():
    print(f"Reading dataset from: {input_file}")
    
    # We will use a dictionary to keep track of counts to see which entities are most common
    # Format: {(entity_string, entity_type): count}
    entity_counts = defaultdict(int)
    
    total_lines = 0
    error_lines = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                
                # Find the assistant's response which contains the JSON array
                assistant_message = None
                for msg in messages:
                    if msg.get("role") == "assistant":
                        assistant_message = msg.get("content", "[]")
                        break
                
                if assistant_message:
                    # Parse the stringified JSON array
                    entities = json.loads(assistant_message)
                    for ent in entities:
                        # Extract the string and the type, convert to lowercase to aggregate duplicates
                        entity_str = ent.get("entity", "").strip().lower()
                        entity_type = ent.get("type", "").strip()
                        
                        if entity_str and entity_type:
                            entity_counts[(entity_str, entity_type)] += 1
                            
            except Exception as e:
                error_lines += 1
                continue

    print(f"Processed {total_lines} total lines. (Errors: {error_lines})")
    print(f"Found {len(entity_counts)} unique entity-type combinations.")
    
    print(f"Saving aggregated entities to: {output_file}")
    
    # Sort by count (descending) so the most common ones are at the top
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['entity', 'type', 'count'])
        for (entity_str, entity_type), count in sorted_entities:
            writer.writerow([entity_str, entity_type, count])
            
    print("Extraction complete!")

if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"Error: Could not find {input_file}")
    else:
        extract_unique_entities()
