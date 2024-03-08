# Load the TorchScript compiled model
model_neuron = torch.jit.load(filename)

# Run inference using the Neuron model
output_neuron = model_neuron(*example)

# Compare the results
print(f"CPU outputs:    {output_cpu[0][0][0][:10]}")
print(f"Neuron outputs: {output_neuron[0][0][0][:10]}")
