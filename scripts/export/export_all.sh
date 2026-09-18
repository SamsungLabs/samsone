#!/bin/bash

set -e

echo "
#########################################################################################################
########################!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!###################################
#########################################################################################################
Before running the script the code below has to be added at .venv/lib/python3.12/site-packages/executorch/examples/models/model_factory.py in the create_model() function to create a text model with embeddings input, not tokens input:
example_kwarg_inputs = {
    'attn_options': {
        'input_pos': torch.tensor([0, 1, 2], dtype=torch.long),
    },
    'h': torch.randn((1,3,576), dtype=torch.float32)),  # tokens, with kv cache our input token length is always just 1 token.,
}
dynamic_shapes = (
        {'input_pos': {0: torch.export.Dim('token_dim', max=500)}},
        {1: torch.export.Dim('token_dim', max=500)},
        
)
example_inputs = model.get_example_inputs()
example_inputs = tuple()
return (
    model.get_eager_model(),
    example_inputs,
    example_kwarg_inputs,
    dynamic_shapes,
)
########################!!!!!!!!!!!!!!!!!and!!!!!!!!!!!!!!!!!!!!#########################################
in .venv/lib/python3.12/site-packages/executorch/examples/models/llama/attention.py at line 486
    change -> if self.enable_dynamic_shape:
    to -> if self.enable_dynamic_shape and len(input_pos)==1:
########################!!!!!!!!!!!!!!!!!and!!!!!!!!!!!!!!!!!!!!#########################################
in .venv/lib/python3.12/site-packages/executorch/examples/models/llama/llama_transformer.py
    change -> freqs_cos, freqs_sin = self.rope.get_freqs(attn_options.get('input_pos'), seqlen)
    to -> freqs_cos, freqs_sin = self.rope.get_freqs_using_indices(attn_options.get('input_pos'))    
########################!!!!!!!!!!!!!!!!!IMPORTANT!!!!!!!!!!!!!!!!!!!!#########################################
"

echo Starting export Samsone99M...
uv run scripts/export/export.py --config-name config99M

echo Starting export Samsone134M...
uv run scripts/export/export.py --config-name config134M

echo Starting export Samsone356M...
uv run scripts/export/export.py --config-name config356M

echo Successfully exported models! You can now use them with the Executorch inference API.
