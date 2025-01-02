import streamlit as st
import pandas as pd
import numpy as np
from integrated_farm_recommendations import (
    predict_yield, get_crop_recommendation, get_fertilizer_recommendation,
    get_pesticide_recommendation, get_water_management_recommendation,
    assess_weather_impact, assess_water_quality,
    irrigation_efficiency,
    crop_water_requirements,
    weather_impact,
    water_quality_parameters,
    initialize_models,
    calculate_water_usage,
    calculate_rotation_score,
    calculate_fertilizer_usage,
    calculate_pesticide_usage
)
import plotly.express as px
import plotly.graph_objects as go

def initialize_models():
    """Initialize all required models"""
    try:
        from integrated_farm_recommendations import train_yield_prediction_model
        yield_model, yield_le, yield_scaler = train_yield_prediction_model()
        
        if yield_model is None or yield_le is None or yield_scaler is None:
            st.error("Failed to initialize one or more models")
            return None
            
        return {
            'yield_model': yield_model,
            'yield_le': yield_le,
            'yield_scaler': yield_scaler
        }
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Sustainable Farming Advisor", layout="wide")
    
    st.title("🌾 Sustainable Farming Recommendation System")
    
    tabs = st.tabs(["Farm Input", "Recommendations", "Analytics"])
    
    # Initialize models at startup
    if 'models' not in st.session_state:
        models = initialize_models()
        if models is not None:
            st.session_state.models = models
        else:
            st.error("Failed to initialize models. Please check your data and model setup.")
            return  # Exit if models aren't properly initialized
    
    # Check if models are properly initialized before proceeding
    if st.session_state.models is None:
        st.error("Models not properly initialized. Please refresh the page or check the setup.")
        return
        
    with tabs[0]:
        st.header("Farm Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            current_crop = st.selectbox(
                "Current Crop",
                ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"]
            )
            
            st.subheader("Previous Crops")
            prev_crop1 = st.selectbox("Previous Crop (1 season ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            prev_crop2 = st.selectbox("Previous Crop (2 seasons ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            prev_crop3 = st.selectbox("Previous Crop (3 seasons ago)", 
                                    ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Potato", "Soybean"])
            
            soil_type = st.selectbox(
                "Soil Type",
                ["Loamy", "Clay", "Sandy", "Silty", "Peaty"]
            )
            
            season = st.selectbox(
                "Season",
                ["Kharif", "Rabi", "Zaid"]
            )
        
        with col2:
            organic_matter = st.slider("Organic Matter Content (%)", 0.0, 30.0, 2.0)
            soil_ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
            
            fertilizer_type = st.multiselect(
                "Current Fertilizer Types",
                ["Urea", "NPK", "DAP", "MOP"]
            )
            
            fertilizer_category = st.selectbox(
                "Fertilizer Category",
                ["Chemical", "Organic", "Mixed"]
            )
            
            irrigation_type = st.selectbox(
                "Irrigation Type",
                ["Drip", "Sprinkler", "Flood", "Manual", "Rain-fed"]
            )
            
            farm_area = st.number_input("Farm Area (acres)", min_value=0.1, value=1.0)
        
        st.subheader("Weather and Water Parameters")
        col3, col4 = st.columns(2)
        
        with col3:
            temperature = st.slider("Current Temperature (°C)", -10.0, 50.0, 25.0)
            rainfall_level = st.selectbox("Rainfall Level", ["Low", "Moderate", "High"])
        
        with col4:
            water_ph = st.slider("Irrigation Water pH", 0.0, 14.0, 7.0)
            salinity_level = st.selectbox("Water Salinity Level", ["Low", "Moderate", "High"])
        
        st.subheader("Pesticide Information")
        current_pesticide = st.selectbox(
            "Current Pesticide Type",
            ["Synthetic Insecticides", "Chemical Fungicides", "Chemical Herbicides"]
        )
        
        pesticide_category = st.selectbox(
            "Pesticide Category",
            ["Chemical", "Organic", "Mixed"]
        )
    
    if st.button("Generate Recommendations"):
        with tabs[1]:
            st.header("Farm Recommendations")
            
            # 1. Crop Rotation
            with st.expander("🌱 Crop Rotation", expanded=True):
                next_crop = get_crop_recommendation(current_crop, 
                                                  [prev_crop1, prev_crop2, prev_crop3],
                                                  season, soil_type)
                unique_crops = len(set([current_crop, prev_crop1, prev_crop2, prev_crop3]))
                rotation_score = (unique_crops / 4) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Recommended next crop:** {next_crop}")
                    st.markdown(f"**Rotation Diversity Score:** {rotation_score:.1f}%")
                    st.progress(rotation_score/100)
                
                with col2:
                    # Crop rotation visualization
                    fig = go.Figure(data=[go.Pie(labels=[current_crop, prev_crop1, prev_crop2, prev_crop3],
                                                hole=.3,
                                                title="Crop History")])
                    st.plotly_chart(fig)
            
            # 2. Fertilizer Recommendations
            with st.expander("🌿 Fertilizer Management", expanded=True):
                fertilizer_recs = get_fertilizer_recommendation(
                    soil_type, current_crop, organic_matter, soil_ph,
                    ", ".join(fertilizer_type), fertilizer_category
                )
                
                # Display recommendations in a clean format
                for rec in fertilizer_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
                
                # Add soil health visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=soil_ph,
                        title={'text': "Soil pH"},
                        gauge={'axis': {'range': [0, 14]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 5.5], 'color': "red"},
                                   {'range': [5.5, 7.5], 'color': "green"},
                                   {'range': [7.5, 14], 'color': "red"}]}))
                    st.plotly_chart(fig)
                with col2:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=organic_matter,
                        title={'text': "Organic Matter (%)"},
                        gauge={'axis': {'range': [0, 30]}}))
                    st.plotly_chart(fig)
            
            # 3. Pesticide Recommendations
            with st.expander("🐛 Pest Management", expanded=True):
                pesticide_recs = get_pesticide_recommendation(
                    current_pesticide, pesticide_category, current_crop, season
                )
                
                for rec in pesticide_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
            
            # 4. Water Management
            with st.expander("💧 Water Management", expanded=True):
                water_recs = get_water_management_recommendation(
                    current_crop, season, soil_type, irrigation_type, farm_area
                )
                
                for rec in water_recs:
                    if rec.startswith('\n'):
                        st.markdown("---")
                    st.markdown(rec)
                
                # Water efficiency visualization
                col1, col2 = st.columns(2)
                with col1:
                    efficiency = irrigation_efficiency.get(irrigation_type, 0.5) * 100
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=efficiency,
                        title={'text': "Irrigation Efficiency (%)"},
                        gauge={'axis': {'range': [0, 100]}}))
                    st.plotly_chart(fig)
                
                with col2:
                    if current_crop in crop_water_requirements and season in crop_water_requirements[current_crop]:
                        water_needed = crop_water_requirements[current_crop][season]
                        fig = go.Figure(go.Indicator(
                            mode="number+delta",
                            value=water_needed,
                            title={'text': "Water Requirement (mm/acre)"}))
                        st.plotly_chart(fig)
            
            # 5. Yield Prediction
            with st.expander("📊 Yield Prediction", expanded=True):
                try:
                    # Calculate required values first
                    water_usage = calculate_water_usage(farm_area, irrigation_type)
                    rotation_score = calculate_rotation_score(current_crop, prev_crop1, prev_crop2, prev_crop3)
                    fertilizer_usage = calculate_fertilizer_usage(farm_area, fertilizer_category)
                    pesticide_usage = calculate_pesticide_usage(farm_area, pesticide_category)
                    
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
                        model=st.session_state.models['yield_model'],
                        le_dict=st.session_state.models['yield_le'],
                        scaler=st.session_state.models['yield_scaler']
                    )
                    
                    # Display predictions only if successful
                    if yield_prediction:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Estimated yield per acre:** {yield_prediction['per_acre']} tons")
                            st.markdown(f"**Total estimated yield:** {yield_prediction['total']} tons")
                            st.markdown(f"**Weather impact factor:** {yield_prediction['weather_impact']}")
                        
                        with col2:
                            # Yield prediction visualization
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=yield_prediction['per_acre'],
                                title={'text': "Yield per Acre (tons)"},
                                gauge={'axis': {'range': [0, 150]},
                                       'bar': {'color': "green"},
                                       'steps': [
                                           {'range': [0, 50], 'color': "lightgray"},
                                           {'range': [50, 100], 'color': "lightgreen"},
                                           {'range': [100, 150], 'color': "darkgreen"}]}
                            ))
                            st.plotly_chart(fig)

                            # Weather impact visualization
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=yield_prediction['weather_impact'] * 100,
                                title={'text': "Weather Impact (%)"},
                                gauge={'axis': {'range': [0, 100]},
                                       'bar': {'color': "blue"},
                                       'steps': [
                                           {'range': [0, 33], 'color': "red"},
                                           {'range': [33, 66], 'color': "yellow"},
                                           {'range': [66, 100], 'color': "green"}]}
                            ))
                            st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error in yield prediction: {str(e)}")
                    st.error("Please check your input data and model setup.")
            
            # Additional Sustainable Practices
            with st.expander("🌍 Sustainable Practices", expanded=True):
                st.markdown("""
                1. Use crop residue as organic matter
                2. Implement mulching
                3. Consider companion planting
                4. Regular soil testing every 6 months
                5. Maintain field borders for beneficial insects
                """)

if __name__ == "__main__":
    main()