import time

# Measure inference time on CPU
start_time_cpu = time.time()
output_cpu = model(*example)
end_time_cpu = time.time()
inference_time_cpu = end_time_cpu - start_time_cpu
print("Inference time on CPU:", inference_time_cpu)

# Measure inference time on Neuron
start_time_neuron = time.time()
output_neuron = model_neuron(*example)
end_time_neuron = time.time()
inference_time_neuron = end_time_neuron - start_time_neuron
print("Inference time on Neuron:", inference_time_neuron)

# Calculate speedup
speedup = inference_time_cpu / inference_time_neuron
print("Neuron is {:.2f} times faster than CPU.".format(speedup))
