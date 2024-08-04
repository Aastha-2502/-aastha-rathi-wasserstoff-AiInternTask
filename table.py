import pandas as pd

def output_table(master_dict):
    rows = []
    master_id = master_dict['MasterImage']['ID']

    for obj in master_dict['MasterImage']['objects']:
        row = {
            'masterID': master_id,
            'objectID': obj['objectID'],
            'object_name': obj['object_type'],
            'extracted_text': obj.get('text', '-'),
            'object_summarised': obj.get('attribute', '-')
        }
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['masterID', 'objectID', 'object_name', 'extracted_text', 'object_summarised'])
    return df
