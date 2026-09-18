from executorch.runtime import Runtime

model_path = "outputs/export/100M/exported/audio_lm.pte"
runtime = Runtime.get()
program = runtime.load_program(model_path)
method_names = program.method_names
print(f"Available methods: {method_names}")

for name in method_names:
    method = program.load_method(name)
    print(method.metadata)
