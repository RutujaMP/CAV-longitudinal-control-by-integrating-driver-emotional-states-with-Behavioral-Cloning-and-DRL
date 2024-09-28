import json
import csv
import os

def convert_json_to_csv(json_dir, csv_file):
    # Initialize CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['config_file', 'level', 'id', 'label'])
        
        # List of all JSON files
        json_files = [pos_json for pos_json in os.listdir(json_dir) if pos_json.startswith('config_') and pos_json.endswith('.json')]

        # Process each JSON file
        for json_file in json_files:
            with open(os.path.join(json_dir, json_file)) as f:
                data = json.load(f)
                config_name = json_file.replace('config_', '').replace('.json', '')

                # Write level_names
                level_names = data.get("level_names", {})
                for level_id, level_name in level_names.items():
                    writer.writerow([config_name, 'level_name', level_id, level_name])

                # Write labels for each level
                for level_id in level_names.keys():
                    labels = data.get(level_id, {})
                    for label_id, label_name in labels.items():
                        writer.writerow([config_name, level_id, label_id, label_name])

    print(f"Annotations have been written to {csv_file}")

# Inputs
json_dir = 'C:/automatic_vehicular_control/datasets/dmd/annotation-tool'  # Update this path to your actual JSON directory
csv_file = 'annotations.csv'  # Desired CSV file name

# Convert JSON to CSV
convert_json_to_csv(json_dir, csv_file)
