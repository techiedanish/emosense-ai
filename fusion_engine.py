# fusion_engine.py

def calculate_fusion(visual_samples, audio_probs, emotion_order):
    """
    Calculates the 70/30 weighted average between visual and audio data.
    """
    if not visual_samples:
        return {emo: 0.0 for emo in emotion_order}

    # 1. Calculate the mean for all visual frames collected in the 3s window
    v_avg = {
        emo: sum(d[emo] for d in visual_samples) / len(visual_samples) 
        for emo in emotion_order
    }

    # 2. Apply Fusion Formula: (70% Visual + 30% Audio)
    fusion_result = {}
    for emo in emotion_order:
        fusion_result[emo] = (v_avg[emo] * 0.7) + (audio_probs[emo] * 0.3)
    
    return fusion_result

def get_dominant_emotion(fusion_probs):
    """
    Identifies the emotion with the highest confidence score.
    """
    if not fusion_probs:
        return "None", 0.0
    
    dom_emo = max(fusion_probs, key=fusion_probs.get)
    return dom_emo, fusion_probs[dom_emo]