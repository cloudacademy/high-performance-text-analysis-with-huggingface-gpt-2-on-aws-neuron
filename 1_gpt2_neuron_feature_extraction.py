import torch
import torch_neuronx
from transformers import GPT2Tokenizer, GPT2Model

# Define the wrapper class
class GPT2Neuron(torch.nn.Module):
    """
    Ensures that `input_ids` and `attention_mask` are passed into the GPT2
    model in the correct positions without requiring a dictionary.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

# Initialize the tokenizer and the GPT2 model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Define the padding token value
gpt2 = GPT2Model.from_pretrained('gpt2', torchscript=True)
model = GPT2Neuron(gpt2)
model.eval()

# Define the text to encode with the tokenizer
text = "Advance your tech career with Cloud Academy."

# Tokenize the text and prepare the example input
encoded_input = tokenizer(
    text,
    max_length=128,            # Specify the maximum length for the inputs
    padding='max_length',      # Pad to the maximum length
    truncation=True,           # Truncate to the maximum length if necessary
    return_tensors='pt'        # Return PyTorch tensors
)

# Prepare the example inputs for the model
example = (
    encoded_input['input_ids'],
    encoded_input['attention_mask'],
)

# Run inference on CPU
output_cpu = model(*example)

# Compile the model using the wrapper
model_neuron = torch_neuronx.trace(model, example)

# Save the compiled moel
filename = 'model.pt'
torch.jit.save(model_neuron, filename)
