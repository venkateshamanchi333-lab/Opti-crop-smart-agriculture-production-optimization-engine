import csv
import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'crop_recommendation.csv')

FEATURES = ['nitrogen', 'phosphorous', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_LABELS = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']

CROP_METADATA = {
    'rice': {
        'display_name': 'Rice',
        'scientific_name': 'Oryza sativa',
        'season': 'Kharif',
        'water_requirement': 'High',
        'growing_duration': '120-150 days',
        'fertilizer': 'Urea, DAP, MOP',
        'market_demand': 'High',
        'difficulty_level': 'Medium',
        'image': 'static/images/rice.svg',
        'harvest_time': 'October-November',
        'common_diseases': 'Blast, Brown Spot',
        'market_value': 'Strong',
        'ideal': {
            'nitrogen': (100, 140),
            'phosphorous': (30, 50),
            'potassium': (30, 50),
            'temperature': (22, 32),
            'humidity': (75, 90),
            'ph': (5.5, 6.5),
            'rainfall': (100, 220)
        },
        'description': 'Rice thrives in warm, humid climates with abundant water supply.',
        'rotation': 'Rotate with legumes or oilseed crops to replenish nitrogen.'
    },
    'maize': {
        'display_name': 'Maize',
        'scientific_name': 'Zea mays',
        'season': 'Kharif/Rabi',
        'water_requirement': 'Medium',
        'growing_duration': '90-120 days',
        'fertilizer': 'NPK blends',
        'market_demand': 'High',
        'difficulty_level': 'Medium',
        'image': 'static/images/maize.svg',
        'harvest_time': 'November-December',
        'common_diseases': 'Stem borer, Rust',
        'market_value': 'High',
        'ideal': {
            'nitrogen': (90, 130),
            'phosphorous': (35, 55),
            'potassium': (25, 45),
            'temperature': (20, 30),
            'humidity': (60, 80),
            'ph': (5.8, 7.0),
            'rainfall': (80, 160)
        },
        'description': 'Maize is an adaptable cereal crop used for food, feed, and industrial production.',
        'rotation': 'Suitable after loamy soils and can follow legumes for better nitrogen.'
    },
    'wheat': {
        'display_name': 'Wheat',
        'scientific_name': 'Triticum aestivum',
        'season': 'Rabi',
        'water_requirement': 'Low',
        'growing_duration': '110-130 days',
        'fertilizer': 'Nitrogen, Phosphorus, Potassium',
        'market_demand': 'Stable',
        'difficulty_level': 'Easy',
        'image': 'static/images/wheat.svg',
        'harvest_time': 'March-April',
        'common_diseases': 'Rust, Blight',
        'market_value': 'Moderate',
        'ideal': {
            'nitrogen': (70, 110),
            'phosphorous': (30, 45),
            'potassium': (20, 35),
            'temperature': (12, 22),
            'humidity': (50, 70),
            'ph': (6.0, 7.5),
            'rainfall': (50, 90)
        },
        'description': 'Wheat performs best in cool, dry weather during its grain filling phase.',
        'rotation': 'Follow with pulses or oilseeds and avoid continuous monoculture.'
    },
    'cotton': {
        'display_name': 'Cotton',
        'scientific_name': 'Gossypium hirsutum',
        'season': 'Kharif',
        'water_requirement': 'Medium',
        'growing_duration': '140-160 days',
        'fertilizer': 'NPK + Micronutrients',
        'market_demand': 'Variable',
        'difficulty_level': 'Medium',
        'image': 'static/images/cotton.svg',
        'harvest_time': 'February-March',
        'common_diseases': 'Bollworm, Jassids',
        'market_value': 'High',
        'ideal': {
            'nitrogen': (80, 120),
            'phosphorous': (30, 55),
            'potassium': (40, 70),
            'temperature': (24, 32),
            'humidity': (55, 75),
            'ph': (5.8, 7.0),
            'rainfall': (40, 90)
        },
        'description': 'Cotton needs warm days and a dry harvest season to produce strong fibers.',
        'rotation': 'Rotate with cereals and legumes to manage pests and soil balance.'
    },
    'sugarcane': {
        'display_name': 'Sugarcane',
        'scientific_name': 'Saccharum officinarum',
        'season': 'Tropical',
        'water_requirement': 'Very High',
        'growing_duration': '240-360 days',
        'fertilizer': 'NPK, Organic Manure',
        'market_demand': 'High',
        'difficulty_level': 'Hard',
        'image': 'static/images/sugarcane.svg',
        'harvest_time': 'October-February',
        'common_diseases': 'Red rot, Smut',
        'market_value': 'Very High',
        'ideal': {
            'nitrogen': (100, 160),
            'phosphorous': (50, 80),
            'potassium': (60, 100),
            'temperature': (24, 34),
            'humidity': (70, 85),
            'ph': (6.0, 7.2),
            'rainfall': (120, 220)
        },
        'description': 'Sugarcane is a high-value, water-intensive crop that prefers fertile soils.',
        'rotation': 'Use as part of a diversified system with legumes to restore nitrogen.'
    }
}


ALL_CROPS = list(CROP_METADATA.keys())


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Learn more by training the model with train_model.py')
    saved = joblib.load(MODEL_PATH)
    model = saved['model'] if isinstance(saved, dict) else saved
    scaler = saved.get('scaler') if isinstance(saved, dict) else None
    encoder = saved.get('encoder') if isinstance(saved, dict) else None
    return model, scaler, encoder


def normalize(value, window_min, window_max, below_good=False):
    if window_min >= window_max:
        return 50
    if below_good:
        return max(0, min(100, int(100 - ((value - window_min) / (window_max - window_min)) * 100))) if value > window_max else 100
    if value < window_min:
        diff = window_min - value
        return max(0, int(100 - (diff / window_min) * 100))
    if value > window_max:
        diff = value - window_max
        return max(0, int(100 - (diff / window_max) * 100))
    return 100


def extract_inputs(form):
    return {
        'nitrogen': float(form.get('nitrogen', 0)),
        'phosphorous': float(form.get('phosphorous', 0)),
        'potassium': float(form.get('potassium', 0)),
        'temperature': float(form.get('temperature', 0)),
        'humidity': float(form.get('humidity', 0)),
        'ph': float(form.get('ph', 0)),
        'rainfall': float(form.get('rainfall', 0)),
        'season': str(form.get('season', 'Any') or 'Any')
    }


def weather_context_for(inputs):
    if inputs['temperature'] >= 30:
        temp_profile = 'Hot'
    elif inputs['temperature'] <= 18:
        temp_profile = 'Cool'
    else:
        temp_profile = 'Moderate'

    if inputs['rainfall'] >= 180:
        rain_profile = 'Wet'
    elif inputs['rainfall'] <= 70:
        rain_profile = 'Dry'
    else:
        rain_profile = 'Balanced'

    return {
        'season': inputs.get('season', 'Any'),
        'temperature_profile': temp_profile,
        'rainfall_profile': rain_profile,
        'profile': f'{temp_profile} and {rain_profile}',
        'regime': 'High rainfall climate' if rain_profile == 'Wet' else 'Low rainfall climate' if rain_profile == 'Dry' else 'Balanced rainfall climate'
    }


def crop_suitability_score(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key, {})
    ideal = crop.get('ideal', {})
    if not ideal:
        return 50
    climate_fields = ['temperature', 'humidity', 'rainfall', 'ph']
    scores = []
    for field in climate_fields:
        target = ideal.get(field)
        if target:
            scores.append(normalize(inputs[field], target[0], target[1]))
    if not scores:
        return 50
    return int(np.mean(scores))


def predict_crop(inputs):
    model, scaler, encoder = load_model()
    vector = np.array([[
        inputs['nitrogen'],
        inputs['phosphorous'],
        inputs['potassium'],
        inputs['temperature'],
        inputs['humidity'],
        inputs['ph'],
        inputs['rainfall']
    ]], dtype=float)
    if scaler is not None:
        vector = scaler.transform(vector)

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(vector)[0]
        index = int(np.argmax(proba))
        crop_key = encoder.inverse_transform([index])[0].lower() if encoder is not None else model.classes_[index].lower()
        confidence = float(np.max(proba))
    else:
        crop_key = model.predict(vector)[0].lower()
        confidence = 0.84

    importance = model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else [1 / len(FEATURES)] * len(FEATURES)
    importance_data = [{'feature': FEATURE_LABELS[i], 'value': round(float(importance[i]) * 100, 1)} for i in range(len(FEATURES))]
    weather_context = weather_context_for(inputs)
    climate_score = crop_suitability_score(inputs, crop_key)
    weather_context['climate_score'] = climate_score
    confidence_score = round(min(99.9, max(40.0, confidence * 100 * 0.7 + climate_score * 0.3)), 2)

    alternatives = []
    for candidate in ALL_CROPS:
        score = crop_suitability_score(inputs, candidate)
        alternatives.append({'crop_key': candidate, 'display_name': CROP_METADATA[candidate].get('display_name', candidate.title()), 'score': score})
    alternatives = sorted(alternatives, key=lambda item: item['score'], reverse=True)[:3]

    return crop_key, confidence_score, importance_data, weather_context, alternatives


def build_recommendation_summary(inputs, crop_key, confidence, weather_context):
    crop = CROP_METADATA.get(crop_key, {})
    display_name = crop.get('display_name', crop_key.title())
    if confidence >= 85 and weather_context.get('climate_score', 0) >= 75:
        headline = f'{display_name} is a strong fit with {confidence:.1f}% confidence.'
        detail = 'The soil and climate inputs align well with the crop’s ideal growth window.'
    elif weather_context.get('climate_score', 0) >= 65:
        headline = f'{display_name} is a good fit for your field conditions.'
        detail = 'The recommendation remains practical, but a few adjustments could improve performance.'
    else:
        headline = f'{display_name} is plausible, but the climate profile is mixed.'
        detail = 'Consider adjusting irrigation, nutrient balance, or selecting a more weather-adapted crop.'
    focus = f'Current weather profile: {weather_context.get("profile", "balanced")} for {weather_context.get("season", "Any")} conditions.'
    return {'headline': headline, 'detail': detail, 'focus': focus}


def explain_prediction(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key, {})
    ideal = crop.get('ideal', {})
    explanations = []
    categories = {
        'nitrogen': 'Nitrogen',
        'phosphorous': 'Phosphorous',
        'potassium': 'Potassium',
        'temperature': 'Temperature',
        'humidity': 'Humidity',
        'ph': 'pH',
        'rainfall': 'Rainfall'
    }
    for field, label in categories.items():
        value = inputs[field]
        target = ideal.get(field)
        if target:
            if target[0] <= value <= target[1]:
                explanations.append(f'{label} of {value} is within the ideal range for {crop.get("display_name")} growth.')
            elif value < target[0]:
                explanations.append(f'{label} of {value} is below the preferred range, so improvements could raise performance.')
            else:
                explanations.append(f'{label} of {value} is above the preferred range, which may affect crop balance.')
        else:
            explanations.append(f'{label} of {value} influences crop health and yield potential.')
    return explanations


def soil_health_scores(inputs):
    score_map = {
        'nitrogen_score': normalize(inputs['nitrogen'], 40, 120),
        'phosphorous_score': normalize(inputs['phosphorous'], 20, 60),
        'potassium_score': normalize(inputs['potassium'], 20, 70),
        'pH_score': normalize(inputs['ph'], 5.8, 7.2),
        'rainfall_score': normalize(inputs['rainfall'], 50, 200),
        'temperature_score': normalize(inputs['temperature'], 18, 34)
    }
    score_map['overall_soil_health'] = int(np.mean(list(score_map.values())))
    suggestions = []
    if score_map['nitrogen_score'] < 65:
        suggestions.append('Increase nitrogen with balanced urea or organic manure.')
    if score_map['phosphorous_score'] < 65:
        suggestions.append('Add rock phosphate or compost to improve phosphorus levels.')
    if score_map['potassium_score'] < 65:
        suggestions.append('Apply potassium sulfate or wood ash for better K balance.')
    if score_map['pH_score'] < 60:
        suggestions.append('Use lime to neutralize acidic soil and improve nutrient uptake.')
    if score_map['rainfall_score'] < 60:
        suggestions.append('Supplement rainfall with targeted irrigation during dry periods.')
    if score_map['temperature_score'] < 60:
        suggestions.append('Consider season selection or frost protection for temperature stress.')
    return score_map, suggestions


def compare_crop_fit(inputs, crop_one, crop_two):
    comparison = []
    for crop_key in [crop_one, crop_two]:
        suitability = calculate_suitability(inputs, crop_key)
        if suitability:
            crop = CROP_METADATA.get(crop_key, {})
            comparison.append({
                'crop_key': crop_key,
                'crop_name': crop.get('display_name', crop_key.title()),
                'score': suitability['score'],
                'status': suitability['status'],
                'water_requirement': crop.get('water_requirement', 'N/A'),
                'season': crop.get('season', 'N/A'),
                'reason': 'Strong fit for current soil and climate profile' if suitability['score'] >= 70 else 'Needs adjustment to match field conditions' if suitability['score'] >= 50 else 'Poor fit for the current conditions'
            })
    return comparison


def calculate_suitability(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key)
    if not crop:
        return None
    ideal = crop['ideal']
    compatibility_components = []
    issues = []
    for key in FEATURES:
        target = ideal.get(key)
        value = inputs[key]
        if target:
            score = normalize(value, target[0], target[1])
            compatibility_components.append(score)
            if score < 65:
                if value < target[0]:
                    issues.append(f'{key.title()} is below the preferred range for {crop.get("display_name")}.')
                else:
                    issues.append(f'{key.title()} is above the preferred range for {crop.get("display_name")}.')
    score = int(np.mean(compatibility_components)) if compatibility_components else 50
    status = 'Compatible' if score >= 70 else 'Marginal' if score >= 50 else 'Not Compatible'
    suggestions = []
    if score < 70:
        suggestions.append('Adjust nutrient balance and irrigation to improve crop suitability.')
    if score < 50:
        suggestions.append('Consider a different crop or invest in soil amendment before planting.')
    return {
        'score': score,
        'status': status,
        'issues': issues,
        'productivity_potential': 'High' if score >= 75 else 'Moderate' if score >= 55 else 'Low',
        'suggestions': suggestions,
        'crop_description': crop.get('description'),
        'crop_rotation': crop.get('rotation')
    }


def yield_prediction(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key)
    if not crop:
        return {'label': 'Unknown', 'score': 0}
    ideal = crop['ideal']
    scores = [normalize(inputs[field], ideal[field][0], ideal[field][1]) for field in FEATURES]
    score = int(np.mean(scores))
    if score >= 80:
        label = 'High'
    elif score >= 60:
        label = 'Medium'
    else:
        label = 'Low'
    trend = ['Low', 'Medium', 'High']
    return {'label': label, 'score': score, 'trend': trend}


def calculate_nutrient_balance(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key)
    ideal = crop['ideal'] if crop else {}
    summary = []
    for component in ['nitrogen', 'phosphorous', 'potassium']:
        if component in ideal:
            value = inputs[component]
            if value < ideal[component][0]:
                summary.append(f'{component.title()} is deficient. Add balanced fertilizer.')
            elif value > ideal[component][1]:
                summary.append(f'{component.title()} is in excess. Reduce fertilizer application.')
            else:
                summary.append(f'{component.title()} is well balanced for {crop.get("display_name")}')
    return summary


def calculate_sustainability_score(inputs, crop_key):
    crop = CROP_METADATA.get(crop_key, {})
    ideal = crop.get('ideal', {})
    water_efficiency = normalize(inputs['rainfall'], ideal.get('rainfall', (60, 150))[0], ideal.get('rainfall', (60, 150))[1])
    fertilizer_efficiency = int(np.clip(100 - ((max(0, inputs['nitrogen'] - 120) + max(0, inputs['phosphorous'] - 60) + max(0, inputs['potassium'] - 80)) / 3), 30, 100))
    eco_score = int(np.mean([water_efficiency, fertilizer_efficiency]))
    carbon_impact = max(0, 100 - eco_score)
    rating = 'Excellent' if eco_score >= 80 else 'Good' if eco_score >= 60 else 'Fair' if eco_score >= 45 else 'Poor'
    return {
        'water_efficiency': water_efficiency,
        'fertilizer_efficiency': fertilizer_efficiency,
        'eco_friendly_score': eco_score,
        'carbon_impact': carbon_impact,
        'rating': rating
    }


def generate_ai_advice(inputs, crop_key):
    advice = []
    if inputs['nitrogen'] < 90:
        advice.append('Add nitrogen-rich organic matter such as green manure or compost.')
    elif inputs['nitrogen'] > 140:
        advice.append('Reduce nitrogen inputs to avoid nutrient burn and environmental runoff.')
    if inputs['ph'] < 6.0:
        advice.append('Apply agricultural lime to raise acidic soil pH.')
    elif inputs['ph'] > 7.5:
        advice.append('Add sulfur or acidic compost to lower alkaline soil pH.')
    if inputs['rainfall'] < 90:
        advice.append('Use drip or sprinkler irrigation to maintain moisture during dry spells.')
    if inputs['humidity'] > 85:
        advice.append('Monitor for fungal diseases and maintain field ventilation to reduce humidity stress.')
    if crop_key in ['rice', 'sugarcane']:
        advice.append('Maintain consistent moisture and avoid water stagnation for these water-intensive crops.')
    else:
        advice.append('Follow crop rotation and integrate organic amendments monthly for soil resilience.')
    return advice


def chatbot_response(message, context=None):
    message = message.lower().strip()
    context = context or {}
    crop_key = context.get('crop_key')
    crop_name = CROP_METADATA.get(crop_key, {}).get('display_name', 'your selected crop') if crop_key else None
    inputs = context.get('inputs') or {}
    weather_context = context.get('weather_context') or {}

    if any(keyword in message for keyword in ['best crop', 'recommend', 'which crop', 'choose']):
        if crop_name:
            return f'Based on your latest inputs, {crop_name} appears promising. I would focus on balanced nutrients, steady irrigation, and monitoring weather stress.'
        return 'Share your soil and weather values and I can suggest the most suitable crop for your field.'
    if any(keyword in message for keyword in ['soil', 'ph', 'nutrient', 'nitrogen', 'phosphorous', 'potassium']):
        if inputs:
            return f'Your current inputs show nitrogen {inputs.get("nitrogen", "N/A")}, phosphorus {inputs.get("phosphorous", "N/A")}, and potassium {inputs.get("potassium", "N/A")}. I would test the soil again if values are extreme and adjust fertilizer carefully.'
        return 'Evaluate soil pH and NPK levels, then apply compost or fertilizers where deficiencies are found to improve soil health.'
    if any(keyword in message for keyword in ['irrigation', 'water', 'rainfall']):
        if weather_context.get('rainfall_profile') == 'Dry':
            return 'Your rainfall pattern looks dry, so a drip irrigation schedule and mulch cover would help preserve soil moisture.'
        return 'Use efficient irrigation such as drip or sprinkler systems and adjust according to rainfall and temperature to avoid waste.'
    if any(keyword in message for keyword in ['pest', 'disease']):
        return 'Practice crop rotation, monitor pests early, and use integrated pest management to protect the crop sustainably.'
    if any(keyword in message for keyword in ['weather', 'temperature', 'humidity']):
        if weather_context:
            return f'Your weather profile looks {weather_context.get("profile", "balanced")}. That suggests you should pay attention to {weather_context.get("regime", "rainfall conditions")} and protect against climate stress.'
        return 'Temperature, humidity, and rainfall strongly influence crop performance, so match your management plan to local weather conditions.'
    if any(keyword in message for keyword in ['yield', 'profit', 'market']):
        return 'For better yield and profitability, prioritize balanced nutrients, good irrigation, and crop rotation based on local conditions.'
    return 'I can help with crop selection, soil health, irrigation, disease prevention, and sustainability planning. Try asking about a specific crop or field issue.'


def load_dataset_stats():
    if not os.path.exists(DATASET_PATH):
        return pd.DataFrame()
    return pd.read_csv(DATASET_PATH)


def flatten_list(values):
    return [str(item) for item in values]


def build_report_context(prediction, inputs, crop_key, explain, feature_importance, soil_scores, advice, yield_estimate, sustainability, suitability):
    return {
        'prediction': prediction,
        'inputs': inputs,
        'crop_key': crop_key,
        'crop_data': CROP_METADATA.get(crop_key, {}),
        'explain': explain,
        'feature_importance': feature_importance,
        'soil_scores': soil_scores,
        'advice': advice,
        'yield_estimate': yield_estimate,
        'sustainability': sustainability,
        'suitability': suitability
    }
