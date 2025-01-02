import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

df = pd.read_csv('sustainable_farming_dataset.csv')

crop_patterns = {
    'Rice': ['Wheat', 'Potato', 'Maize'],
    'Wheat': ['Rice', 'Soybean', 'Maize'],
    'Cotton': ['Wheat', 'Maize', 'Soybean'],
    'Maize': ['Wheat', 'Soybean', 'Rice'],
    'Sugarcane': ['Wheat', 'Soybean', 'Potato'],
    'Potato': ['Rice', 'Maize', 'Wheat'],
    'Soybean': ['Wheat', 'Rice', 'Maize']
}

fertilizer_mapping = {
    'Chemical': {
        'Urea': {'organic': 'Vermicompost', 'transition_time': '3-6 months'},
        'NPK': {'organic': 'Bio-fertilizer', 'transition_time': '4-6 months'},
        'DAP': {'organic': 'Compost', 'transition_time': '3-4 months'},
        'MOP': {'organic': 'Green Manure', 'transition_time': '2-3 months'}
    },
    'Mixed': {
        'Urea + Compost': {'organic': 'Compost', 'transition_time': '2-3 months'},
        'NPK + Bio-fertilizer': {'organic': 'Bio-fertilizer', 'transition_time': '1-2 months'},
        'DAP + Vermicompost': {'organic': 'Vermicompost', 'transition_time': '1-2 months'}
    }
}

pesticide_mapping = {
    'Chemical': {
        'Synthetic Insecticides': {'organic': 'Neem Oil', 'transition_time': '2-3 months'},
        'Chemical Fungicides': {'organic': 'Trichoderma', 'transition_time': '2-4 months'},
        'Chemical Herbicides': {'organic': 'Mulching + Manual Weeding', 'transition_time': '1-2 months'}
    },
    'Mixed': {
        'Limited Chemical + Neem': {'organic': 'Neem Oil', 'transition_time': '1-2 months'},
        'Integrated Pest Management': {'organic': 'Continue IPM', 'transition_time': 'Already sustainable'}
    }
}

crop_water_requirements = {
    'Rice': {'Kharif': 1200, 'Rabi': 1000, 'Zaid': 1400},
    'Wheat': {'Kharif': 450, 'Rabi': 400, 'Zaid': 500},
    'Cotton': {'Kharif': 700, 'Rabi': 650, 'Zaid': 800},
    'Maize': {'Kharif': 500, 'Rabi': 450, 'Zaid': 600},
    'Sugarcane': {'Kharif': 1500, 'Rabi': 1400, 'Zaid': 1700},
    'Potato': {'Kharif': 500, 'Rabi': 450, 'Zaid': 550},
    'Soybean': {'Kharif': 450, 'Rabi': 400, 'Zaid': 500}
}

irrigation_efficiency = {
    'Drip': 0.9,
    'Sprinkler': 0.75,
    'Flood': 0.6,
    'Manual': 0.5,
    'Rain-fed': 0.4
}

weather_impact = {
    'Temperature': {
        'Optimal': {'Rice': (25, 35), 'Wheat': (20, 25), 'Cotton': (21, 35), 
                   'Maize': (20, 30), 'Sugarcane': (25, 35), 'Potato': (15, 25), 
                   'Soybean': (20, 30)},
        'Impact': {'Low': 0.7, 'Optimal': 1.0, 'High': 0.8}
    },
    'Rainfall': {
        'Low': 0.8,
        'Moderate': 1.0,
        'High': 0.9
    }
}

water_quality_parameters = {
    'pH': {
        'Optimal': (6.5, 7.5),
        'Impact': {'Low': 0.8, 'Optimal': 1.0, 'High': 0.85}
    },
    'Salinity': {
        'Low': {'level': '<1000 ppm', 'impact': 1.0},
        'Moderate': {'level': '1000-2000 ppm', 'impact': 0.9},
        'High': {'level': '>2000 ppm', 'impact': 0.7}
    }
}

def get_crop_recommendation(current_crop, prev_crops, season, soil_type):
    if current_crop in crop_patterns:
        return np.random.choice(crop_patterns[current_crop])
    return "No specific recommendation available"

def get_fertilizer_recommendation(soil_type, current_crop, organic_matter, soil_ph, 
                                current_fertilizer, fertilizer_category):
    recommendations = []
    
    # Soil health recommendations
    if organic_matter < 2:
        recommendations.append("Low organic matter content:")
        recommendations.append("- Add compost or vermicompost")
        recommendations.append("- Consider green manuring")
    
    if soil_ph < 5.5:
        recommendations.append("\nAcidic soil conditions:")
        recommendations.append("- Add lime to increase pH")
        recommendations.append("- Use pH tolerant organic fertilizers")
    elif soil_ph > 7.5:
        recommendations.append("\nAlkaline soil conditions:")
        recommendations.append("- Add organic matter to balance pH")
        recommendations.append("- Consider sulfur application")
    
    # Fertilizer transition recommendations
    if fertilizer_category in fertilizer_mapping:
        if current_fertilizer in fertilizer_mapping[fertilizer_category]:
            organic_alt = fertilizer_mapping[fertilizer_category][current_fertilizer]
            recommendations.append(f"\nFertilizer transition plan:")
            recommendations.append(f"- Current: {current_fertilizer}")
            recommendations.append(f"- Recommended: {organic_alt['organic']}")
            recommendations.append(f"- Transition time: {organic_alt['transition_time']}")
    
    return recommendations

def get_pesticide_recommendation(current_pesticide, pesticide_category, crop, season):
    recommendations = []
    
    if pesticide_category in pesticide_mapping:
        if current_pesticide in pesticide_mapping[pesticide_category]:
            organic_alt = pesticide_mapping[pesticide_category][current_pesticide]
            recommendations.append(f"\nPesticide transition plan:")
            recommendations.append(f"- Current: {current_pesticide}")
            recommendations.append(f"- Recommended: {organic_alt['organic']}")
            recommendations.append(f"- Transition time: {organic_alt['transition_time']}")
    
    recommendations.append("\nIntegrated Pest Management (IPM) Practices:")
    recommendations.append("- Use pest monitoring and thresholds")
    recommendations.append("- Implement biological control methods")
    recommendations.append("- Practice crop rotation for pest management")
    recommendations.append("- Use companion planting for natural pest control")
    
    return recommendations

def train_yield_prediction_model():
    """Train model to predict crop yield"""
    try:
        df = pd.read_csv('sustainable_farming_dataset.csv')
        print("Dataset loaded successfully")
        
        # Separate categorical and numerical features
        categorical_features = [
            'Current_Crop',
            'Soil_Type',
            'Season',
            'Fertilizer_Category',
            'Irrigation_Type'
        ]
        
        numerical_features = [
            'Organic_Matter_Content(%)',
            'Soil_pH',
            'Water_Usage(cubic meters)',
            'Rotation_Health_Score',
            'Fertilizer_Used(tons)',
            'Pesticide_Used(kg)'
        ]
        
        # Initialize transformers
        le_dict = {}  # Dictionary to store a LabelEncoder for each categorical column
        
        # Encode each categorical column separately
        for col in categorical_features:
            le_dict[col] = LabelEncoder()
            df[col] = le_dict[col].fit_transform(df[col])
        
        # Scale numerical features
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        
        # Combine features
        X = df[categorical_features + numerical_features]
        y = df['Yield(tons)']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Print model performance
        train_score = model.score(X, y)
        print(f"Model R² score: {train_score:.3f}")
        
        return model, le_dict, scaler
        
    except Exception as e:
        print(f"Error in training model: {str(e)}")
        print(f"Available columns in dataset: {df.columns.tolist()}")
        raise

def train_crop_recommendation_model():
    """Train model to recommend next crop"""
    try:
        df = pd.read_csv('sustainable_farming_dataset.csv')
        
        # Use exact column names from dataset
        features = [
            'Current_Crop',  # Changed from current_crop
            'Previous_Crop_1',
            'Previous_Crop_2',
            'Previous_Crop_3',
            'Soil_Type',
            'Season',
            'Organic_Matter_Content(%)',
            'Soil_pH',
            'Fertilizer_Category',
            'Pesticide_Category',
            'Irrigation_Type'
        ]
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = [
            'Current_Crop',
            'Previous_Crop_1',
            'Previous_Crop_2',
            'Previous_Crop_3',
            'Soil_Type',
            'Season',
            'Fertilizer_Category',
            'Pesticide_Category',
            'Irrigation_Type'
        ]
        
        # Create a copy of the dataframe to avoid modifying the original
        X = df[features].copy()
        y = df['Current_Crop'].copy()  # Target is current crop to predict next crop
        
        # Encode categorical variables
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col])
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, le
        
    except Exception as e:
        print(f"Error in training crop recommendation model: {str(e)}")
        print(f"Available columns in dataset: {df.columns.tolist()}")
        raise

def calculate_rotation_score(current_crop, prev1, prev2, prev3):
    """Calculate rotation health score based on crop diversity"""
    crops = [current_crop, prev1, prev2, prev3]
    unique_crops = len(set(crops))
    # Score from 0-100 based on crop diversity
    return (unique_crops / 4) * 100

def calculate_water_usage(farm_area, irrigation_type):
    """Estimate water usage based on irrigation type and farm area"""
    # Base water usage per acre (cubic meters)
    base_usage = {
        'Drip': 3000,
        'Sprinkler': 4000,
        'Flood': 6000,
        'Manual': 4500,
        'Rain-fed': 2000
    }
    return base_usage.get(irrigation_type, 4000) * farm_area

def calculate_fertilizer_usage(farm_area, category):
    """Estimate fertilizer usage based on farm area and category"""
    # Base fertilizer usage per acre (tons)
    base_usage = {
        'Chemical': 0.5,
        'Organic': 1.2,
        'Mixed': 0.8
    }
    return base_usage.get(category, 0.8) * farm_area

def calculate_pesticide_usage(farm_area, category):
    """Estimate pesticide usage based on farm area and category"""
    # Base pesticide usage per acre (kg)
    base_usage = {
        'Chemical': 2.0,
        'Organic': 3.0,
        'Mixed': 2.5
    }
    return base_usage.get(category, 2.5) * farm_area

def calculate_weather_impact(temperature, rainfall_level):
    """Calculate weather impact on yield (0-1 scale)"""
    # Optimal temperature range for most crops
    optimal_temp_min = 15
    optimal_temp_max = 30
    
    # Temperature impact (0-1)
    if optimal_temp_min <= temperature <= optimal_temp_max:
        temp_impact = 1.0
    else:
        temp_impact = max(0, 1 - abs(temperature - optimal_temp_max) / 20)
    
    # Rainfall impact (0-1)
    rainfall_impact = {
        'Low': 0.6,
        'Moderate': 1.0,
        'High': 0.8
    }
    
    # Combined impact
    weather_impact = (temp_impact + rainfall_impact.get(rainfall_level, 0.7)) / 2
    return round(weather_impact, 2)

def predict_yield(current_crop, soil_type, season, organic_matter, soil_ph,
                 fertilizer_category, irrigation_type, farm_area,
                 water_usage, rotation_score, fertilizer_usage, pesticide_usage,
                 temperature, rainfall_level,  # Add these parameters
                 model, le_dict, scaler):
    """Predict yield based on input parameters"""
    try:
        # Create input DataFrame with numerical values
        input_data = pd.DataFrame({
            'Current_Crop': [current_crop],
            'Soil_Type': [soil_type],
            'Season': [season],
            'Fertilizer_Category': [fertilizer_category],
            'Irrigation_Type': [irrigation_type],
            'Organic_Matter_Content(%)': [float(organic_matter)],
            'Soil_pH': [float(soil_ph)],
            'Water_Usage(cubic meters)': [float(water_usage)],
            'Rotation_Health_Score': [float(rotation_score)],
            'Fertilizer_Used(tons)': [float(fertilizer_usage)],
            'Pesticide_Used(kg)': [float(pesticide_usage)]
        })
        
        # Encode categorical variables
        categorical_cols = [
            'Current_Crop',
            'Soil_Type',
            'Season',
            'Fertilizer_Category',
            'Irrigation_Type'
        ]
        
        for col in categorical_cols:
            input_data[col] = le_dict[col].transform(input_data[col])
        
        # Scale numerical features
        numerical_cols = [
            'Organic_Matter_Content(%)',
            'Soil_pH',
            'Water_Usage(cubic meters)',
            'Rotation_Health_Score',
            'Fertilizer_Used(tons)',
            'Pesticide_Used(kg)'
        ]
        
        input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
        
        # Calculate weather impact
        weather_impact = calculate_weather_impact(temperature, rainfall_level)
        
        # Adjust yield prediction based on weather impact
        predicted_yield = model.predict(input_data)[0] * weather_impact
        
        return {
            'per_acre': round(predicted_yield, 2),
            'total': round(predicted_yield * farm_area, 2),
            'weather_impact': weather_impact
        }
        
    except Exception as e:
        print(f"Error in yield prediction: {str(e)}")
        if 'input_data' in locals():
            print(f"Input data types: {input_data.dtypes}")
        raise

def get_water_management_recommendation(crop, season, soil_type, irrigation_type, farm_area):
    """Generate water management recommendations"""
    
    recommendations = []
    
  
    if crop in crop_water_requirements and season in crop_water_requirements[crop]:
        water_needed = crop_water_requirements[crop][season]
        efficiency = irrigation_efficiency.get(irrigation_type, 0.7)
        

        total_water_needed = (water_needed * farm_area) / efficiency
        
        recommendations.append(f"\nWater Management Plan:")
        recommendations.append(f"- Base water requirement: {water_needed} mm/acre")
        recommendations.append(f"- Irrigation efficiency ({irrigation_type}): {efficiency*100}%")
        recommendations.append(f"- Total water needed: {total_water_needed:.2f} mm for {farm_area} acres")
        
        
        if irrigation_type == 'Drip':
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Daily light irrigation")
            recommendations.append(f"- Approximately {(water_needed/30):.1f} mm/day")
        elif irrigation_type == 'Sprinkler':
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Irrigate every 2-3 days")
            recommendations.append(f"- Approximately {(water_needed/15):.1f} mm per session")
        else:
            recommendations.append("\nRecommended irrigation schedule:")
            recommendations.append("- Irrigate every 5-7 days")
            recommendations.append(f"- Approximately {(water_needed/6):.1f} mm per session")
        
        
        recommendations.append("\nWater conservation measures:")
        recommendations.append("- Use mulching to reduce evaporation")
        recommendations.append("- Monitor soil moisture regularly")
        recommendations.append("- Irrigate during early morning or evening")
        
        if irrigation_type not in ['Drip', 'Sprinkler']:
            recommendations.append("\nSuggested improvements:")
            recommendations.append("- Consider upgrading to drip irrigation")
            recommendations.append("- Install soil moisture sensors")
            recommendations.append("- Implement rainfall harvesting")
    
    return recommendations

def assess_weather_impact(crop, temperature, rainfall_level):
    """Assess impact of weather conditions on yield"""
    impact_factors = []
    recommendations = []
    
   
    optimal_temp = weather_impact['Temperature']['Optimal'].get(crop, (20, 30))
    if temperature < optimal_temp[0]:
        temp_impact = weather_impact['Temperature']['Impact']['Low']
        recommendations.append(f"Temperature below optimal range for {crop}")
        recommendations.append("- Consider cold protection measures")
        recommendations.append("- Adjust planting time to warmer period")
    elif temperature > optimal_temp[1]:
        temp_impact = weather_impact['Temperature']['Impact']['High']
        recommendations.append(f"Temperature above optimal range for {crop}")
        recommendations.append("- Consider shade protection")
        recommendations.append("- Increase irrigation frequency")
    else:
        temp_impact = weather_impact['Temperature']['Impact']['Optimal']
        recommendations.append("Temperature in optimal range")
    
    
    rainfall_impact = weather_impact['Rainfall'].get(rainfall_level, 1.0)
    if rainfall_level == 'Low':
        recommendations.append("Low rainfall conditions:")
        recommendations.append("- Implement water conservation measures")
        recommendations.append("- Consider drought-resistant varieties")
    elif rainfall_level == 'High':
        recommendations.append("High rainfall conditions:")
        recommendations.append("- Ensure proper drainage")
        recommendations.append("- Monitor for disease pressure")
    
    return temp_impact * rainfall_impact, recommendations

def assess_water_quality(water_ph, salinity_level):
    """Assess impact of water quality on irrigation"""
    recommendations = []
    
    
    if water_ph < water_quality_parameters['pH']['Optimal'][0]:
        ph_impact = water_quality_parameters['pH']['Impact']['Low']
        recommendations.append("Low water pH:")
        recommendations.append("- Consider pH adjustment")
        recommendations.append("- Monitor soil pH regularly")
    elif water_ph > water_quality_parameters['pH']['Optimal'][1]:
        ph_impact = water_quality_parameters['pH']['Impact']['High']
        recommendations.append("High water pH:")
        recommendations.append("- Add acidifying agents to irrigation water")
        recommendations.append("- Monitor soil pH regularly")
    else:
        ph_impact = water_quality_parameters['pH']['Impact']['Optimal']
        recommendations.append("Water pH in optimal range")
    
    
    salinity_impact = water_quality_parameters['Salinity'][salinity_level]['impact']
    if salinity_level != 'Low':
        recommendations.append(f"\nWater salinity ({salinity_level}):")
        recommendations.append("- Monitor soil salinity")
        recommendations.append("- Consider salt-tolerant crops")
        if salinity_level == 'High':
            recommendations.append("- Implement leaching practices")
            recommendations.append("- Increase irrigation frequency")
    
    return ph_impact * salinity_impact, recommendations

def initialize_models():
    """Initialize and return trained models"""
    yield_model, yield_le_dict, yield_scaler = train_yield_prediction_model()
    crop_model, crop_le = train_crop_recommendation_model()
    return {
        'yield_model': yield_model,
        'yield_le_dict': yield_le_dict,
        'yield_scaler': yield_scaler,
        'crop_model': crop_model,
        'crop_le': crop_le
    }

def main():
    print("\n=== Integrated Sustainable Farming Recommendation System ===\n")
    
    
    current_crop = input("Enter current crop: ").capitalize()
    prev_crop1 = input("Enter previous crop (1 season ago): ").capitalize()
    prev_crop2 = input("Enter previous crop (2 seasons ago): ").capitalize()
    prev_crop3 = input("Enter previous crop (3 seasons ago): ").capitalize()
    
    print("\nSoil Types: Loamy, Clay, Sandy, Silty, Peaty")
    soil_type = input("Enter soil type: ").capitalize()
    
    print("\nSeasons: Kharif, Rabi, Zaid")
    season = input("Enter season: ").capitalize()
    
    
    organic_matter = float(input("\nEnter organic matter content (%): "))
    soil_ph = float(input("Enter soil pH: "))
    
    
    print("\nCurrent Fertilizer Types: Urea, NPK, DAP, MOP")
    current_fertilizer = input("Enter current fertilizer type: ").upper()
    
    print("\nFertilizer Categories: Chemical, Organic, Mixed")
    fertilizer_category = input("Enter fertilizer category: ").capitalize()
    
    
    print("\nCurrent Pesticide Types: Synthetic Insecticides, Chemical Fungicides, Chemical Herbicides")
    current_pesticide = input("Enter current pesticide type: ").title()
    
    print("\nPesticide Categories: Chemical, Organic, Mixed")
    pesticide_category = input("Enter pesticide category: ").capitalize()
    
    
    print("\nIrrigation Types: Drip, Sprinkler, Flood, Manual, Rain-fed")
    irrigation_type = input("Enter irrigation type: ").capitalize()
    
    farm_area = float(input("Enter farm area (acres): "))
    
    
    temperature = float(input("\nEnter current temperature (°C): "))
    print("\nRainfall Levels: Low, Moderate, High")
    rainfall_level = input("Enter rainfall level: ").capitalize()
    
    water_ph = float(input("\nEnter irrigation water pH: "))
    print("\nWater Salinity Levels: Low, Moderate, High")
    salinity_level = input("Enter water salinity level: ").capitalize()
    
    
    print("\n=== Comprehensive Farm Recommendations ===")
    
    
    next_crop = get_crop_recommendation(current_crop, 
                                      [prev_crop1, prev_crop2, prev_crop3],
                                      season, soil_type)
    print("\n1. Crop Rotation Recommendation:")
    print(f"- Recommended next crop: {next_crop}")
    
    
    unique_crops = len(set([current_crop, prev_crop1, prev_crop2, prev_crop3]))
    rotation_score = (unique_crops / 4) * 100
    print(f"- Rotation Diversity Score: {rotation_score:.1f}%")
    
   
    print("\n2. Fertilizer Recommendations:")
    fertilizer_recs = get_fertilizer_recommendation(
        soil_type, current_crop, organic_matter, soil_ph, 
        current_fertilizer, fertilizer_category
    )
    for rec in fertilizer_recs:
        print(rec)
    
  
    print("\n3. Pesticide Recommendations:")
    pesticide_recs = get_pesticide_recommendation(
        current_pesticide, pesticide_category, current_crop, season
    )
    for rec in pesticide_recs:
        print(rec)
    

    print("\n4. Yield Prediction:")
    # Initialize models for command-line usage
    models = initialize_models()
    
    # Calculate numerical values
    rotation_score = calculate_rotation_score(
        current_crop, prev_crop1, prev_crop2, prev_crop3
    )
    
    water_usage = calculate_water_usage(farm_area, irrigation_type)
    
    fertilizer_usage = calculate_fertilizer_usage(farm_area, fertilizer_category)
    
    pesticide_usage = calculate_pesticide_usage(farm_area, pesticide_category)
    
    # Make yield prediction
    yield_prediction = predict_yield(
        current_crop=current_crop,
        soil_type=soil_type,
        season=season,
        organic_matter=organic_matter,
        soil_ph=soil_ph,
        fertilizer_category=fertilizer_category,
        irrigation_type=irrigation_type,
        farm_area=farm_area,
        water_usage=water_usage,
        rotation_score=rotation_score,
        fertilizer_usage=fertilizer_usage,
        pesticide_usage=pesticide_usage,
        temperature=temperature,
        rainfall_level=rainfall_level,
        model=models['yield_model'],
        le_dict=models['yield_le_dict'],
        scaler=models['yield_scaler']
    )
    print(f"- Estimated yield per acre: {yield_prediction['per_acre']} tons")
    print(f"- Total estimated yield: {yield_prediction['total']} tons")
    print(f"- Weather impact factor: {yield_prediction['weather_impact']}")
    print(f"  (Weather impact considers temperature and rainfall conditions)")
    
    _, weather_recs = assess_weather_impact(current_crop, temperature, rainfall_level)
    print("\nWeather-based Recommendations:")
    for rec in weather_recs:
        print(rec)

    _, water_quality_recs = assess_water_quality(water_ph, salinity_level)
    print("\nWater Quality Recommendations:")
    for rec in water_quality_recs:
        print(rec)
    
    # 5. Water Management
    print("\n5. Water Management:")
    water_recs = get_water_management_recommendation(
        current_crop, season, soil_type, irrigation_type, farm_area
    )
    for rec in water_recs:
        print(rec)
    
    print("\nAdditional Sustainable Practices:")
    print("1. Use crop residue as organic matter")
    print("2. Implement mulching")
    print("3. Consider companion planting")
    print("4. Regular soil testing every 6 months")
    print("5. Maintain field borders for beneficial insects")

if __name__ == "__main__":
    try:
        while True:
            main()
            if input("\nWould you like another recommendation? (yes/no): ").lower() != 'yes':
                break
    except KeyboardInterrupt:
        print("\nThank you for using the recommendation system!")