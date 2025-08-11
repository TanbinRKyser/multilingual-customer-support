import torch
from captum.attr import IntegratedGradients
from intent_classifier.bert_infer import model, tokenizer, device

model.to( device ).eval()

def forward_func( input_ids, attention_mask ):
    return model( input_ids = input_ids, attention_mask = attention_mask ).logits

ig = IntegratedGradients( forward_func )

def explain_intent_ig( text: str ):
    enc = tokenizer( text, return_tensors = "pt", truncation = True, padding = True )
    enc = { k: v.to( device ) for k, v in enc.items() }

    with torch.no_grad():
        pred_class = model( **enc ).logits.argmax( dim = 1 ).item()

    attributions = ig.attribute(
        inputs = enc[ "input_ids" ],
        additional_forward_args = ( enc[ "attention_mask" ], ),
        target = pred_class,
        n_steps = 32,
        return_convergence_delta = True
    )

    scores = attributions.sum( dim = -1 ).squeeze( 0 ).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens( enc[ "input_ids" ].squeeze( 0 ).detach().cpu().numpy() )


    ## normalize the filters
    out = []
    max_score = max( abs( s ) for s in scores ) or 1.0 
    for t,s in zip( tokens, scores ):
        if t not in ( "[CLS]", "[SEP]", "[PAD]" ):
            out.append( {"token": t, "weight": float( s / max_score ) } )
    
    return out