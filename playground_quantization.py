#このスクリプトは、量子化し計測する前にその量子化を動くように試行錯誤するためのスクリプト
import argparse
import torch
from eagle.model.ea_model import EaModel
import re
import os
from torchao.quantization.quant_api import quantize_, Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
from torchao.sparsity.sparse_api import sparsify_,SemiSparseWeightConfig
from torchao.dtypes import SemiSparseLayout

def quantization(draft_model,library,data_type,group_size,packing_format=None):
    # Quantization
    if library == "torchao":
        # [TorchAO](https://github.com/pytorch/ao)
        # See https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#quantization-techniques
        from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
        from torchao.quantization.quantize_.workflows import Int4PackingFormat
        if data_type == "int2":
            raise ValueError("Unsupported data type")
        elif data_type == "int4":
            if packing_format == "tile_packed_to_4d":
                torch.to(model, dtype=torch.bfloat16)
                int4_packing_format = Int4PackingFormat.TILE_PACKED_TO_4D
            elif packing_format == "marlin_sparse":
                int4_packing_format = Int4PackingFormat.MARLIN_SPARSE
            else:
                assert False, f"Unsupported int4 packing format(on A100): {packing_format}"
            
            if group_size=='None':
                quantize_(draft_model, Int4WeightOnlyConfig(int4_packing_format=int4_packing_format))
            elif group_size.isdigit():
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(group_size), int4_packing_format=int4_packing_format))
            elif re.search(r'block\d+', group_size) is not None:# e.g.,5120+13824 最初の数字はfilter5120用、次の数字はfilter13824用のグループサイズ
                blocks = re.findall(r'block(\d+)', group_size)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(blocks[0]))), filter_fn= filter5120, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(10240/int(blocks[0]))), filter_fn=filter10240, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(15360/int(blocks[0]))), filter_fn=filter15360, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(13824/int(blocks[0]))), filter_fn=filter13824, int4_packing_format=int4_packing_format)
            elif re.search(r'tansaku', group_size) is not None:
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(2))), filter_fn= filter5120notlmhead, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(1))), filter_fn= filter5120lmhead, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(10240/int(2))), filter_fn=filter10240, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(15360/int(2))), filter_fn=filter15360, int4_packing_format=int4_packing_format)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(13824/int(2))), filter_fn=filter13824, int4_packing_format=int4_packing_format) 
            else:
                assert False
        elif data_type == "int8":
            if group_size=='None':
                quantize_(draft_model, Int8WeightOnlyConfig())
            elif group_size.isdigit():
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(group_size)))
            elif re.search(r'block\d+', group_size) is not None:# e.g.,5120+13824 最初の数字はfilter5120用、次の数字はfilter13824用のグループサイズ
                blocks = re.findall(r'block(\d+)', group_size)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int( 5120/int(blocks[0]))), filter_fn= filter5120)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(10240/int(blocks[0]))), filter_fn=filter10240)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(15360/int(blocks[0]))), filter_fn=filter15360)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(13824/int(blocks[0]))), filter_fn=filter13824)
            elif re.search(r'tansaku', group_size) is not None:
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int( 5120/int(2))), filter_fn= filter5120notlmhead)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int( 5120/int(1))), filter_fn= filter5120lmhead)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(10240/int(2))), filter_fn=filter10240)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(15360/int(2))), filter_fn=filter15360)
                quantize_(draft_model, Int8WeightOnlyConfig(group_size=int(13824/int(2))), filter_fn=filter13824)
            else:
                assert False
        elif data_type == "W8A8":
            quantize_(draft_model, Int8DynamicActivationInt8WeightConfig(),filter_fn=filtermiddlelayer)
        else:
            assert False
    elif library == "optimum-quanto":
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        # [Optimum Quanto](https://github.com/huggingface/optimum-quanto)
        # See [Quantization workflow for vanilla pytorch models (low-level API)](https://github.com/huggingface/optimum-quanto?tab=readme-ov-file#quantization-workflow-for-vanilla-pytorch-models-low-level-api)
        from optimum.quanto import quantize, freeze, qint2, qint4, qint8
        if data_type == "int2":
            qdraft_model = quantize(draft_model, weights=qint2)
            print(qdraft_model)
        elif data_type == "int4":
            quantize(draft_model, weights=qint4)
            freeze(draft_model)
            print(draft_model)
        elif data_type == "int8":
            qdraft_model = quantize(draft_model, weights=qint8)
            draft_model = qdraft_model
            print(draft_model)
        else:
            assert False
    else:
        assert False

def cal_model_memory_bytes(model):
    from torchao.dtypes import AffineQuantizedTensor
    from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import Int4TilePackedTo4dTensor

    total_bytes = 0
    for p in model.parameters():
        if isinstance(p, AffineQuantizedTensor):
            # 量子化された重み
            total_bytes += p.tensor_impl.data.numel() * p.tensor_impl.data.element_size()
            #スケール
            total_bytes += p.tensor_impl.scale.numel() * p.tensor_impl.scale.element_size()
            #ゼロポイント
            total_bytes += p.tensor_impl.zero_point.numel() * p.tensor_impl.zero_point.element_size()
        
        elif isinstance(p, Int4TilePackedTo4dTensor):
            #qdata
            total_bytes += p.qdata.numel() * p.qdata.element_size()
            #scale_and_zero
            total_bytes += p.scale_and_zero.numel() * p.scale_and_zero.element_size()
            #act_pre_scale ある場合のみ計算 活性値のスケールを行う
            if hasattr(p, 'act_pre_scale') and p.act_pre_scale is not None:
                total_bytes += p.act_pre_scale.numel() * p.act_pre_scale.element_size()
            
        else:
            # 量子化されてない時
            total_bytes += p.numel() * p.element_size()
    return total_bytes

def filter5120(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120

def filter5120notlmhead(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120 and 'lm_head' not in fqn

def filter5120lmhead(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120 and 'lm_head' in fqn

def filtermiddlelayer(module: torch.nn.Module, fqn: str) -> bool:
    #lmheadと最後のfc層以外のLinear層を対象にするフィルタ
    return isinstance(module, torch.nn.Linear) and 'lm_head' not in fqn and 'fc' not in fqn

def filter10240(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが10240ののLinear層を対象にするフィルタ 10240=1024*10
    return isinstance(module, torch.nn.Linear) and module.in_features == 10240

def filter15360(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが15360ののLinear層を対象にするフィルタ 15360=1024*15
    return isinstance(module, torch.nn.Linear) and module.in_features == 15360

def filter13824(module: torch.nn.Module, fqn: str) -> bool:
    #in_features=13824のLinear層を対象にするフィルタ 13824=512*27
    return isinstance(module, torch.nn.Linear) and module.in_features == 13824

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--library", choices=["torchao", "optimum-quanto"], default="torchao", help="quantization library to use (default: %(default)s)")
    parser.add_argument("-d", "--data-type", choices=["int2", "int4", "int8"], default="int8", help="quantization data type (default: %(default)s)")
    args = parser.parse_args()

    # Loading
    model = EaModel.from_pretrained(
        base_model_path="lmsys/vicuna-13b-v1.3",
        ea_model_path="yuhuili/EAGLE3-Vicuna1.3-13B",
        torch_dtype="auto",
        device_map="cuda")
    model.eval().cuda()
    target_model = model.base_model
    draft_model = model.ea_layer
    print(f'{cal_model_memory_bytes(draft_model)/(1024*1024):.2f} MB in EA layer before quantization.')
    #draft_model.to(dtype=torch.bfloat16)
    #draft_model.to(device="cuda:0")
    #quantize_(model, Int8DynamicActivationInt8WeightConfig())
    #model=torch.compile(model, mode="max-autotune")
    #quantize_(model, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()))

    quantization(draft_model, 'torchao', 'W8A8', group_size='128', packing_format='marlin_sparse')
    #draft_model.to(device="cuda:0")
    print(f'{cal_model_memory_bytes(draft_model)/(1024*1024):.2f} MB in EA layer after quantization.')
    print(draft_model)
    # Copied from EAGLE/README.md
    from fastchat.model import get_conversation_template
    your_message="Hello"
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], your_message)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids=model.tokenizer([prompt]).input_ids
    input_ids = torch.as_tensor(input_ids).cuda()
    output_ids=model.eagenerate(input_ids,temperature=0.5,max_new_tokens=512)
    output=model.tokenizer.decode(output_ids[0])

    print(output)