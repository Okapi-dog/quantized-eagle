import json
def cal_acc(file_path):
    data = []
    acc = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    for item in data:
        #len_turnsを足しているのは、idxsはforループで何回目なので、通常+1しなければならない。また、turns毎にforループは初期化されるため。
        acc.append(round(sum(item['choices'][0]['new_tokens']) / (sum(item['choices'][0]['idxs'])+len(item['choices'][0]['turns'])), 3))
    return acc

def cal_time(file_path):
    data = []
    draft_forward_times = []
    target_forward_times = []
    wall_times = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    for item in data:
        draft_forward_times.append(sum(item['choices'][0]['draft_forward_time'])/(sum(item['choices'][0]['idxs'])+len(item['choices'][0]['turns'])))
        target_forward_times.append(sum(item['choices'][0]['target_forward_time'])/(sum(item['choices'][0]['idxs'])+len(item['choices'][0]['turns'])))
        wall_times.append(sum(item['choices'][0]['wall_time'])/(sum(item['choices'][0]['idxs'])+len(item['choices'][0]['turns'])))
    return draft_forward_times, target_forward_times, wall_times

if __name__ == "__main__":
    acc =cal_acc('data/vicuna-13b/original_gsNone_total60_d10_k10_tmp1.0_kb0_pb0.0.jsonl')
    print(sum(acc)/len(acc))
    draft_forward_times, target_forward_times, wall_times = cal_time('data/vicuna-13b/original_gsNone_total60_d10_k10_tmp1.0_kb0_pb0.0.jsonl')
    print(sum(draft_forward_times)/len(draft_forward_times))
    print(sum(target_forward_times)/len(target_forward_times))
    print(sum(wall_times)/len(wall_times))
    print(sum(draft_forward_times)/len(draft_forward_times)+sum(target_forward_times)/len(target_forward_times))