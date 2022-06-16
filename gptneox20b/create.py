import json
import os
from tqdm import auto as tqdm_lib
from os.path import exists

import accelerate
import torch
import tokenizers

import gptneox20b.model as model20b
from gptneox20b.constants import Args20b, ArgsDummy

def create_index(model, index_path):
    no_split_module_classes = ['TransformerLayer']
    index = accelerate.infer_auto_device_map(
        model=model,
        no_split_module_classes=no_split_module_classes
    )
    for key, value in index.items():
        if(key[:10] == 'layer_list'):
            layer = int(float(key[11:13])) + 2
        elif(key == 'embed_in'):
            layer = 0
        elif(key == 'final_layer_norm'):
            layer = 47
        elif(key == 'logits_out'):
            layer = 48

        index[key] = f"layer_{layer:02d}.pt"

    with open(index_path, "w") as f:
        json.dump(index, f, indent=4)

def create_model(checkpoint_path, use_cache=False, device=torch.device("cuda:0")):
    """
    To prevent allocation memory on CPU, we initialize on 'meta' and individually
    port each module over to 'device' as we load each state dict.

    :param checkpoint_path: Path to the checkpoint folder
    :param use_cache: whether to use cache (i.e. for efficient generation)
    :param device: device that you want the model to end up on
    :return: model
    """
    if not exists("./cache"):
        create_cache(checkpoint_path)

    # Instantiate model
    print("Instantiating model")
    with accelerate.init_empty_weights():
        model = model20b.NeoX20BModel(Args20b, use_cache=use_cache)
        # model = model.half().to_empty(device=device)

    no_split_module_classes = ['TransformerLayer']
    max_memory = {'cuda:0': 18*(2**30), 'cpu': 38*(2**30) }
    full_model_device_map = accelerate.infer_auto_device_map(
        model=model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes
    )
    print(full_model_device_map)

    accelerate.load_checkpoint_and_dispatch(
        model=model, 
        checkpoint='./cache/.index.json', 
        device_map=full_model_device_map, 
        offload_folder='./cache/offload',
        offload_buffers=True,
        no_split_module_classes=no_split_module_classes,
        dtype='float16'
    )
    return model

def create_cache(checkpoint_path):
    pbar = tqdm_lib.tqdm(total=48)
    pbar.set_description("Creating cache")
    with accelerate.init_empty_weights():
        model = model20b.NeoX20BModel(Args20b)
    pbar.update(1)

    os.mkdir("./cache")

    # Helpers to initialize a module and pass it to accelerate
    def cache_state_dict(layer, module, state_dict):

        if layer == 0:
            parent = 'embed_in'
        elif layer == 47:
            parent = 'final_layer_norm'
        elif layer == 48:
            parent = 'logits_out'
        else:
            parent = f'layer_list.{layer-2:d}'

        rooted_dict = {}
        for key, value in state_dict.items():
            rooted_key = f"{parent:s}.{key:s}"
            rooted_dict[rooted_key] = value

        path = f"./cache/layer_{layer:02d}.pt"
        torch.save(obj=rooted_dict, f=path)

        return module
        device_map = accelerate.infer_auto_device_map(module)
        #print(device_map)
        offload_folder = f'./cache/offload/layer_{layer:02d}'

        accelerate.load_checkpoint_in_model(
            model=module, 
            checkpoint=path, 
            device_map=device_map, 
            offload_folder=offload_folder,
            dtype='float16'
            )
        accelerate.dispatch_model(
            model=module, 
            device_map=device_map, 
            main_device=device,
            offload_dir=offload_folder
            )
        #module.load_state_dict(state_dict)
        #device_map = { name:device for name, _ in module.state_dict().items() }
        #accelerate.dispatch_model(module, device_map=device_map, main_device=device)
        return module

    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
        loaded_tp1 = torch.load(os.path.join(checkpoint_path, filename_tp1))
        loaded_tp2 = torch.load(os.path.join(checkpoint_path, filename_tp2))
        state_dict = {}
        # Good
        # Keys where we concatenate on the second dim
        for key in [
            "attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
        ]:
            state_dict[key] = torch.cat([loaded_tp1[key], loaded_tp2[key]], dim=1)
        # Mapping individual split weights to custom split implementations
        # Layer Norms
        # Choose 1
        state_dict["input_layernorm.weight"] = (
            loaded_tp1["input_layernorm.weight"] + loaded_tp2["input_layernorm.weight"]) / 2
        state_dict["input_layernorm.bias"] = (
            loaded_tp1["input_layernorm.bias"] + loaded_tp2["input_layernorm.bias"]) / 2
        state_dict["post_attention_layernorm.weight"] = (
            loaded_tp1["post_attention_layernorm.weight"] + loaded_tp2["post_attention_layernorm.weight"]) / 2
        state_dict["post_attention_layernorm.bias"] = (
            loaded_tp1["post_attention_layernorm.bias"] + loaded_tp2["post_attention_layernorm.bias"]) / 2
        # LinearWithTPMerge
        state_dict["mlp.dense_h_to_4h.weight"] = torch.cat([
            loaded_tp1["mlp.dense_h_to_4h.weight"],
            loaded_tp2["mlp.dense_h_to_4h.weight"],
        ], dim=0)
        state_dict["mlp.dense_h_to_4h.bias"] = torch.cat([
            loaded_tp1["mlp.dense_h_to_4h.bias"],
            loaded_tp2["mlp.dense_h_to_4h.bias"],
        ], dim=0)
        state_dict["attention.query_key_value.weight"] = torch.cat([
            loaded_tp1["attention.query_key_value.weight"],
            loaded_tp2["attention.query_key_value.weight"],
        ], dim=0)
        state_dict["attention.query_key_value.bias"] = torch.cat([
            loaded_tp1["attention.query_key_value.bias"],
            loaded_tp2["attention.query_key_value.bias"],
        ], dim=0)
        # LinearWithTPSplitBias
        state_dict["mlp.dense_4h_to_h.bias"] = (
            loaded_tp1["mlp.dense_4h_to_h.bias"]
            + loaded_tp2["mlp.dense_4h_to_h.bias"]
        )
        state_dict["attention.dense.bias"] = (
            loaded_tp1["attention.dense.bias"]
            + loaded_tp2["attention.dense.bias"]
        )
        # Just take one
        state_dict["attention.rotary_emb.inv_freq"] = loaded_tp1["attention.rotary_emb.inv_freq"]
        # model.layer_list[layer_i].load_state_dict(state_dict)
        model.layer_list[layer_i] = cache_state_dict(layer_i+2, model.layer_list[layer_i], state_dict)
        del loaded_tp1
        del loaded_tp2
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Loading input embedding")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    # model.embed_in.load_state_dict({"weight": torch.cat([
    #     loaded_tp1["word_embeddings.weight"],
    #     loaded_tp2["word_embeddings.weight"],
    # ], dim=0)})
    state_dict = {"weight": torch.cat([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], dim=0)}
    model.embed_in = cache_state_dict(0, model.embed_in, state_dict)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Loading final layer norm")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    # model.final_layer_norm.load_state_dict({
    #     "weight": (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"])/2,
    #     "bias": (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"])/2,
    # })
    state_dict = {
        "weight": (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"])/2,
        "bias": (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"])/2,
    }
    model.final_layer_norm = cache_state_dict(47, model.final_layer_norm, state_dict)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Loading output embedding")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    # model.logits_out.load_state_dict({
    #     "weight": torch.cat([
    #         loaded_tp1["final_linear.weight"],
    #         loaded_tp2["final_linear.weight"],
    #     ], dim=0),
    # })
    state_dict = {
        "weight": torch.cat([
            loaded_tp1["final_linear.weight"],
            loaded_tp2["final_linear.weight"],
        ], dim=0),
    }
    model.logits_out = cache_state_dict(48, model.logits_out, state_dict)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)
    pbar.set_description("Done.")

    create_index(model, './cache/.index.json')

    return model


def create_dummy_model(use_cache=False, device=torch.device("cpu")):
    model = model20b.NeoX20BModel(ArgsDummy, use_cache=use_cache).half().to(device)
    return model


def create_tokenizer(tokenizer_path):
    return tokenizers.Tokenizer.from_file(tokenizer_path)
