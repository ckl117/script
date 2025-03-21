from safetensors.numpy import save_file as numpy_save_file
import os

from paddlenlp.utils.safetensors import fast_load_file as numpy_load_file

files_idx_list = [str(i) for i in range(160, 164)]
model_name_template = "model-00{}-of-000163.safetensors"

input_dir = "/root/.paddlenlp/models/deepseek-ai/DeepSeek-R1-FP8"
# output_model = "/root/.paddlenlp/models/yuhuili/EAGLE-llama2-chat-7B-rename/model.safetensors"
output_model = "/root/.paddlenlp/models/deepseek-ai/DeepSeek-R1-MTP-FP8/model.safetensors"

new_model_state = {}
for file_idx in files_idx_list:
    input_model_path = os.path.join(input_dir, model_name_template.format(file_idx))

    tensors = numpy_load_file(input_model_path)

    for k in list(tensors.keys()):
        # if "lm_head.weight" in k:
        #     new_model_state[k] = tensors[k]
        if "layers.61" in k:
            if "embed_tokens" in k:
                new_name = "deepseek_v3.embed_tokens.weight"
            elif "shared_head.norm.weight" in k:
                new_name = "deepseek_v3.norm.weight"
            elif "shared_head.head.weight" in k:
                new_name = "lm_head.weight"
            elif "enorm.weight" in k:
                new_name = "deepseek_v3.enorm.weight"
            elif "hnorm.weight" in k:
                new_name = "deepseek_v3.hnorm.weight"
            elif "eh_proj.weight" in k:
                new_name = "deepseek_v3.eh_proj.weight"
            else:
                new_name = k.replace("layers.61.", "layers.0.")
                # new_model_state[k.replace("61.", "0.")] = tensors[k]
            new_name = new_name.replace("deepseek_v3", "deepseek_v3_mtp")
            print(f"{k} ---> {new_name}")
            new_model_state[new_name] = tensors.pop(k)
    print("new_model_state.keys()", new_model_state.keys())

# import pdb;pdb.set_trace()
# numpy_save_file(new_model_state, output_model, metadata={"format": "np"})

print("sss")


