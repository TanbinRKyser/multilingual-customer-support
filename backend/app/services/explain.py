from lime.lime_text import LimeTextExplainer
import numpy as np

def dummy_predict( texts ):
    return np.array([ [0.3,0.7] for _ in texts ])

def explain_input_text( text ):
    
    explainer = LimeTextExplainer( class_names = ['negative', 'positive'] )
    explanation = explainer.explain_instance( text, dummy_predict, num_features = 5 )

    return explanation.as_list() 