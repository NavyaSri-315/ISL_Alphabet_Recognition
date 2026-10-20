import pandas as pd
import numpy as np

def preprocess_data(input_csv='isl_landmarks.csv', output_csv='isl_landmarks_processed.csv'):
    print("Loading dataset for preprocessing...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: {input_csv} not found. Run 'extract_features.py' first!")
        return

    df = df.dropna()
    labels = df['label']
    features = df.drop(columns=['label'])
    
    processed_data = []
    
    print("Applying normalization...")
    for index, row in features.iterrows():
        coords = row.values.reshape(21, 2)
        
        # Translation: Shift relative to wrist (Landmark 0)
        wrist = coords[0]
        normalized_coords = coords - wrist
        
        # Scale: Normalize by max distance from wrist
        max_dist = np.max(np.sqrt(np.sum(normalized_coords**2, axis=1)))
        if max_dist > 0:
            normalized_coords = normalized_coords / max_dist
            
        processed_data.append(normalized_coords.flatten())
        
    columns = []
    for i in range(21):
        columns.append(f'x_{i}')
        columns.append(f'y_{i}')
        
    processed_df = pd.DataFrame(processed_data, columns=columns)
    processed_df['label'] = labels.values
    
    cols = ['label'] + [col for col in processed_df.columns if col != 'label']
    processed_df = processed_df[cols]
    
    processed_df.to_csv(output_csv, index=False)
    print(f"Preprocessing complete! Saved to '{output_csv}'.")

if __name__ == "__main__":
    preprocess_data()