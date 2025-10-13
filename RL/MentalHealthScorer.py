from transformers import pipeline
import torch

class MentalHealthScorer:
    def __init__(self):
        self.emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if torch.cuda.is_available() else -1,
            trust_remote_code=True
        )
        
        self.empathy_keywords = [
            'understand', 'feel', 'hearing', 'difficult', 'challenging',
            'appreciate', 'recognize', 'valid', 'sense', 'experience',
            'going through', 'must be', 'sounds like'
        ]
        
        self.supportive_keywords = [
            'help', 'support', 'here for you', 'together', 'work through',
            'cope', 'manage', 'handle', 'strategy', 'approach'
        ]
        
        self.harmful_keywords = [
            'just get over it', 'not a big deal', 'overreacting',
            'snap out of it', 'attention seeking', 'being dramatic'
        ]
        
        self.professional_keywords = [
            'therapist', 'professional', 'doctor', 'counselor',
            'mental health professional', 'seek help', 'crisis line'
        ]
    
    def score_length(self, text):
        word_count = len(text.split())
        if 50 <= word_count <= 250:
            return 1.0
        elif word_count < 50:
            return 0.3  
        elif word_count > 300:
            return 0.7  
        else:
            return 0.8
    
    def score_empathy(self, text):
        text_lower = text.lower()
        empathy_count = sum(1 for keyword in self.empathy_keywords if keyword in text_lower)
        # normalize
        return min(empathy_count / 3.0, 1.0)
    
    def score_supportiveness(self, text):
        text_lower = text.lower()
        support_count = sum(1 for keyword in self.supportive_keywords if keyword in text_lower)
        # normalize
        return min(support_count / 2.0, 1.0)
    
    def score_safety(self, text):
        text_lower = text.lower()
        harmful_count = sum(1 for keyword in self.harmful_keywords if keyword in text_lower)
        if harmful_count > 0:
            return 0.0 
        return 1.0
    
    def score_professionalism(self, text):
        text_lower = text.lower()
        professional_count = sum(1 for keyword in self.professional_keywords if keyword in text_lower)
        if professional_count > 0:
            return 1.0
        return 0.5  
    
    def score_emotion_appropriateness(self, text):
        result = self.emotion_classifier(text)[0] 
        label = result['label'].lower()
        score = result['score']
        
        if label in ['joy', 'neutral', 'surprise']:
            return score
        elif label in ['sadness']:
            return score * 0.8
        elif label in ['fear', 'anger', 'disgust']:
            return score * 0.3  
        else:
            return 0.5
    
    def calculate_quality_score(self, text):
        scores = {
            'length': self.score_length(text),
            'empathy': self.score_empathy(text),
            'supportiveness': self.score_supportiveness(text),
            'safety': self.score_safety(text),
            'professionalism': self.score_professionalism(text),
            'emotion': self.score_emotion_appropriateness(text),
        }
        
        weights = {
            'length': 0.1,
            'empathy': 0.20,
            'supportiveness': 0.20,
            'safety': 0.30,  
            'professionalism': 0.15,
            'emotion': 0.15,
        }
        
        final_score = sum(scores[k] * weights[k] for k in scores.keys())
        return final_score
