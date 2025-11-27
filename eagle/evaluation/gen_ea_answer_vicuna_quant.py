"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
from torchao.dtypes import AffineQuantizedTensor
import re
import torch

try:
    from ..model.ea_model import EaModel
    from ..model.kv_cache import initialize_past_key_values
    from ..model.utils import *
except:
    from eagle.model.ea_model import EaModel
    from eagle.model.kv_cache import initialize_past_key_values
    from eagle.model.utils import *



def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    print(f"Total Number of questions: {len(questions)}")
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = EaModel.from_pretrained(
        base_model_path=base_model_path,
        ea_model_path=ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        use_eagle3=args.use_eagle3,
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)
    
    target_model = model.base_model
    draft_model = model.ea_layer
    before_memory = cal_model_memory_bytes(draft_model)/(1024*1024)
    #一つのGPUに収まるようにする
    #draft_model.to(device="cuda:0")
    quantization(draft_model, args.library, args.data_type, args.group_size, packing_format=args.packing_format)
    #draft_model.to(device="cuda:0")
    after_memory = cal_model_memory_bytes(draft_model)/(1024*1024)


    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    if not args.nowarmup:
        for _ in range(3):
            torch.manual_seed(0)

            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                # try:
                torch.cuda.synchronize()
                start_time = time.perf_counter()

                output_ids, new_token, idx, _, _ = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True
                )
                torch.cuda.synchronize()
                total_time = time.perf_counter() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]
                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                conv.stop_str = "</s>"
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
        print('Warmup done')
    else:
        print('No warmup')

    # questions=questions[6:]
    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            target_forward_time = []
            draft_forward_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids


                torch.cuda.synchronize()
                start_time = time.perf_counter()
                output_ids, new_token, idx, target_time, draft_time = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    top_p=args.top_p_basemodel,
                    top_k=args.top_k_basemodel,
                    log=True
                )
                torch.cuda.synchronize()
                total_time = time.perf_counter() - start_time
                output_ids = output_ids[0][len(input_ids[0]):]

                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()


                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                target_forward_time.append(target_time)
                draft_forward_time.append(draft_time)
                conv.messages[-1][-1] = output
            # torch.cuda.empty_cache()
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time, "target_forward_time": target_forward_time, "draft_forward_time": draft_forward_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
                "before_memory": before_memory,
                "after_memory": after_memory
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])

def filter5120(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120

def filter5120notlmhead(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120 and 'lm_head' not in fqn

def filter5120lmhead(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが5120ののLinear層を対象にするフィルタ 5120=1024*5
    return isinstance(module, torch.nn.Linear) and module.in_features == 5120 and 'lm_head' in fqn

def filter10240(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが10240ののLinear層を対象にするフィルタ 10240=1024*10
    return isinstance(module, torch.nn.Linear) and module.in_features == 10240

def filter15360(module: torch.nn.Module, fqn: str) -> bool:
    #in_featuresが15360ののLinear層を対象にするフィルタ 15360=1024*15
    return isinstance(module, torch.nn.Linear) and module.in_features == 15360

def filter13824(module: torch.nn.Module, fqn: str) -> bool:
    #in_features=13824のLinear層を対象にするフィルタ 13824=512*27
    return isinstance(module, torch.nn.Linear) and module.in_features == 13824

def filtermiddlelayer(module: torch.nn.Module, fqn: str) -> bool:
    #lmheadと最後のfc層以外のLinear層を対象にするフィルタ
    return isinstance(module, torch.nn.Linear) and 'lm_head' not in fqn and 'fc' not in fqn

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

def quantization(draft_model,library,data_type,group_size,packing_format=None):
    # Quantization
    if library == "torchao":
        # [TorchAO](https://github.com/pytorch/ao)
        # See https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#quantization-techniques
        from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig
        from torchao.quantization.quantize_.workflows import Int4PackingFormat
        from torchao.dtypes import SemiSparseLayout
        if data_type == "int2":
            raise ValueError("Unsupported data type")
        elif data_type == "int4":
            if packing_format == "tile_packed_to_4d":
                int4_packing_format = Int4PackingFormat.TILE_PACKED_TO_4D
                draft_model.to(dtype=torch.bfloat16)
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
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(blocks[0])), int4_packing_format=int4_packing_format), filter_fn= filter5120)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(10240/int(blocks[0])), int4_packing_format=int4_packing_format), filter_fn=filter10240)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(15360/int(blocks[0])), int4_packing_format=int4_packing_format), filter_fn=filter15360)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(13824/int(blocks[0])), int4_packing_format=int4_packing_format), filter_fn=filter13824)
            elif re.search(r'tansaku', group_size) is not None:
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(2)), int4_packing_format=int4_packing_format), filter_fn= filter5120notlmhead)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int( 5120/int(1)), int4_packing_format=int4_packing_format), filter_fn= filter5120lmhead)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(10240/int(2)), int4_packing_format=int4_packing_format), filter_fn=filter10240)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(15360/int(2)), int4_packing_format=int4_packing_format), filter_fn=filter15360)
                quantize_(draft_model, Int4WeightOnlyConfig(group_size=int(13824/int(2)), int4_packing_format=int4_packing_format), filter_fn=filter13824) 
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
            torch._dynamo.config.cache_size_limit = 64
            quantize_(draft_model, Int8DynamicActivationInt8WeightConfig(layout=SemiSparseLayout()), filter_fn=filtermiddlelayer)
            #opt_forward = torch.compile(draft_model.forward, mode="default", dynamic=True)
            #draft_model.forward = opt_forward
        
        elif data_type == "original":
            pass
        else:
            assert False
    elif library == "optimum-quanto":
        # [Optimum Quanto](https://github.com/huggingface/optimum-quanto)
        # See [Quantization workflow for vanilla pytorch models (low-level API)](https://github.com/huggingface/optimum-quanto?tab=readme-ov-file#quantization-workflow-for-vanilla-pytorch-models-low-level-api)
        from optimum.quanto import quantize, freeze, qint2, qint4, qint8
        if data_type == "int2":
            quantize(draft_model, weights=qint2)
        elif data_type == "int4":
            quantize(draft_model, weights=qint4)
        elif data_type == "int8":
            quantize(draft_model, weights=qint8)
        else:
            assert False
        freeze(draft_model)
    else:
        assert False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/v-yuhuili/b/res/v13/h0/checkpoints/state_1/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/v-yuhuili/b/weights/vicuna/13B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-vicuna-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--nowarmup", action="store_true",
        help="No warmup before evaluation.",
    )

    parser.add_argument(
        "--use-eagle3",
        action="store_true"
    )
    parser.add_argument(
        "-l", "--library", choices=["torchao", "optimum-quanto"], default="torchao",
        help="quantization library to use (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--data-type", choices=["int2", "int4", "int8","W8A8", "original"], default="original",
        help="quantization data type (default: %(default)s)"
    )
    parser.add_argument(
        "--group_size", type=str, default='None',
        help="quantization grop size (defalt: per channnel quantization)"
    )
    parser.add_argument(
        "--packing_format", type=str, default='tile_packed_to_4d',
        help="quantization packing format (default: %(default)s)"
    )
    parser.add_argument(
        "--top-p-basemodel", type=float, default=0.0,
        help="top-p for base model sampling (default: %(default)s)"
    )
    parser.add_argument(
        "--top-k-basemodel", type=int, default=0,
        help="top-k for base model sampling (default: %(default)s)"
    )

    args = parser.parse_args()

    for k,v in vars(args).items():
        print(f"{k}={v}")

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"{args.bench_name}/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

    reorg_answer_file(answer_file)
