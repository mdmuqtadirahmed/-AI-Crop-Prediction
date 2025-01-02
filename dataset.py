import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def generate_sustainable_farming_dataset(n_samples=4500):
    # Reference values from existing dataset
    crop_types = ['Rice', 'Wheat', 'Cotton', 'Maize', 'Sugarcane', 'Potato', 'Soybean', 'Tomato', 'Carrot', 'Barley']
    soil_types = ['Loamy', 'Clay', 'Sandy', 'Silty', 'Peaty']
    irrigation_types = ['Sprinkler', 'Drip', 'Flood', 'Rain-fed', 'Manual']
    seasons = ['Kharif', 'Rabi', 'Zaid']

    # Define healthy crop rotation patterns
    healthy_rotations = {
        'Rice': ['Wheat', 'Legumes', 'Maize'],
        'Wheat': ['Rice', 'Soybean', 'Potato'],
        'Cotton': ['Wheat', 'Soybean', 'Maize'],
        'Maize': ['Wheat', 'Soybean', 'Potato'],
        'Sugarcane': ['Soybean', 'Wheat', 'Potato'],
        'Potato': ['Maize', 'Wheat', 'Soybean'],
        'Soybean': ['Wheat', 'Maize', 'Rice'],
        'Tomato': ['Maize', 'Soybean', 'Wheat'],
        'Carrot': ['Maize', 'Soybean', 'Wheat'],
        'Barley': ['Soybean', 'Potato', 'Maize']
    }

    # Generate base data
    data = {
        'Farm_ID': [f'F{str(i).zfill(4)}' for i in range(n_samples)],
        'Current_Crop': np.random.choice(crop_types, n_samples),
        'Farm_Area(acres)': np.random.uniform(10, 500, n_samples),
        'Irrigation_Type': np.random.choice(irrigation_types, n_samples),
        'Soil_Type': np.random.choice(soil_types, n_samples),
        'Season': np.random.choice(seasons, n_samples)
    }

    df = pd.DataFrame(data)

    # Add rotation history (last 3 seasons)
    def generate_crop_history(current_crop):
        # 70% chance of following good rotation, 30% chance of poor rotation
        if np.random.random() < 0.7:
            # Good rotation
            return np.random.choice(healthy_rotations[current_crop], 3, replace=False)
        else:
            # Poor rotation (same crop repeated)
            return [current_crop] * 3

    df['Previous_Crop_1'] = df['Current_Crop'].apply(lambda x: generate_crop_history(x)[0])
    df['Previous_Crop_2'] = df['Current_Crop'].apply(lambda x: generate_crop_history(x)[1])
    df['Previous_Crop_3'] = df['Current_Crop'].apply(lambda x: generate_crop_history(x)[2])

    # Calculate rotation health score
    def calculate_rotation_score(row):
        crops = [row['Current_Crop'], row['Previous_Crop_1'], 
                row['Previous_Crop_2'], row['Previous_Crop_3']]
        unique_crops = len(set(crops))
        legume_count = sum(1 for crop in crops if crop in ['Soybean', 'Legumes'])
        
        # Score based on diversity and legume inclusion
        diversity_score = (unique_crops / 4) * 70  # Max 70 points
        legume_score = (legume_count / 3) * 30  # Max 30 points
        return diversity_score + legume_score

    df['Rotation_Health_Score'] = df.apply(calculate_rotation_score, axis=1)

    # Add realistic correlations
    # Farm area influences other metrics
    df['Fertilizer_Used(tons)'] = (
        df['Farm_Area(acres)'] * np.random.uniform(0.01, 0.03, n_samples) + 
        np.random.normal(0, 0.5, n_samples)
    ).clip(0.5, 10)

    df['Pesticide_Used(kg)'] = (
        df['Farm_Area(acres)'] * np.random.uniform(0.005, 0.015, n_samples) + 
        np.random.normal(0, 0.3, n_samples)
    ).clip(0.1, 5)

    # Water usage based on irrigation type and area
    irrigation_multipliers = {
        'Flood': 300,
        'Sprinkler': 200,
        'Drip': 150,
        'Rain-fed': 100,
        'Manual': 250
    }
    
    df['Water_Usage(cubic meters)'] = df.apply(
        lambda x: x['Farm_Area(acres)'] * irrigation_multipliers[x['Irrigation_Type']] * 
        (1 + np.random.uniform(-0.2, 0.2)), axis=1
    )

    # Yield based on multiple factors
    base_yields = {
        'Rice': 30,
        'Wheat': 25,
        'Cotton': 15,
        'Maize': 35,
        'Sugarcane': 40,
        'Potato': 25,
        'Soybean': 20,
        'Tomato': 45,
        'Carrot': 35,
        'Barley': 20
    }

    df['Yield(tons)'] = df.apply(
        lambda x: (
            base_yields[x['Current_Crop']] * 
            (x['Farm_Area(acres)']/100) * 
            (1 + np.random.uniform(-0.3, 0.3))
        ), axis=1
    )

    # Add sustainability indicators
    df['Organic_Matter_Content(%)'] = np.random.uniform(1, 5, n_samples)
    df['Soil_pH'] = np.random.uniform(5.5, 7.5, n_samples)
    df['Chemical_Free_Days'] = np.random.randint(0, 365, n_samples)
    
    # Calculate sustainability score
    df['Sustainability_Score'] = (
        (df['Water_Usage(cubic meters)'].rank(pct=True) * -1 + 1) * 30 +  # Lower water usage is better
        (df['Fertilizer_Used(tons)'].rank(pct=True) * -1 + 1) * 30 +     # Lower fertilizer usage is better
        (df['Pesticide_Used(kg)'].rank(pct=True) * -1 + 1) * 40          # Lower pesticide usage is better
    )

    # Round numeric columns to 2 decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(2)

    # Define fertilizer and pesticide types
    fertilizer_types = {
        'Chemical': ['Urea', 'NPK', 'DAP', 'MOP'],
        'Organic': ['Compost', 'Vermicompost', 'Green Manure', 'Bio-fertilizer'],
        'Mixed': ['Urea + Compost', 'NPK + Bio-fertilizer', 'DAP + Vermicompost']
    }
    
    pesticide_types = {
        'Chemical': ['Synthetic Insecticides', 'Chemical Fungicides', 'Chemical Herbicides'],
        'Organic': ['Neem Extract', 'Biological Control', 'Herbal Pesticides'],
        'Mixed': ['Limited Chemical + Neem', 'Integrated Pest Management']
    }
    
    # Add to existing dataframe structure
    df['Fertilizer_Category'] = np.random.choice(['Chemical', 'Organic', 'Mixed'], 
                                               n_samples, p=[0.6, 0.2, 0.2])
    df['Pesticide_Category'] = np.random.choice(['Chemical', 'Organic', 'Mixed'], 
                                              n_samples, p=[0.7, 0.15, 0.15])
    
    # Add specific types based on category
    df['Current_Fertilizer'] = df['Fertilizer_Category'].apply(
        lambda x: np.random.choice(fertilizer_types[x])
    )
    df['Current_Pesticide'] = df['Pesticide_Category'].apply(
        lambda x: np.random.choice(pesticide_types[x])
    )
    
    # Add sustainable alternatives
    def suggest_sustainable_alternative(row):
        if row['Fertilizer_Category'] == 'Chemical':
            return {
                'fertilizer': np.random.choice(fertilizer_types['Organic']),
                'transition_time': np.random.choice(['3 months', '6 months']),
                'expected_benefit': np.random.choice([
                    'Improved soil health', 
                    'Better water retention',
                    'Enhanced soil microbiome'
                ])
            }
        return {'fertilizer': row['Current_Fertilizer'], 
                'transition_time': 'Already sustainable',
                'expected_benefit': 'Maintaining good practices'}
    
    df['Sustainable_Recommendations'] = df.apply(suggest_sustainable_alternative, axis=1)
    
    return df

# Generate dataset
sustainable_farming_df = generate_sustainable_farming_dataset()

# Save to CSV
sustainable_farming_df.to_csv('sustainable_farming_dataset.csv', index=False)

# Display sample and basic statistics
print("Dataset Shape:", sustainable_farming_df.shape)
print("\nSample of the dataset:")
print(sustainable_farming_df.head())
print("\nBasic Statistics:")
print(sustainable_farming_df.describe())
