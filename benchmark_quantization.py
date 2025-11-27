import subprocess
import matplotlib.pyplot as plt
import os 

def get_accdata(filestr,qdata_type: str,
                group_size=32,
                total_token:int=60,
                depth:int=5,
                top_k:int=10,
                temperature:float=1.0,
                top_k_basemodel:int=0,
                top_p_basemodel:float=0.0):

    savefile = f'./{filestr}{qdata_type}_gs{group_size}_total{total_token}_d{depth}_k{top_k}_tmp{temperature}_kb{top_k_basemodel}_pb{top_p_basemodel}.jsonl'
    if os.path.exists(savefile):
        print(f'File {savefile} exists, skip generation.')
    else:
        subprocess.run(['python3', '-m', 'eagle.evaluation.gen_ea_answer_vicuna_quant',
                    '--ea-model-path', 'yuhuili/EAGLE3-Vicuna1.3-13B',
                    '--base-model-path', 'lmsys/vicuna-13b-v1.3',
                    '--use-eagle3',
                    '--answer-file', savefile,
                    '--question-end', '10',
                    '--data-type', qdata_type,
                    '--group_size', str(group_size) ,
                    '--packing_format', 'tile_packed_to_4d',
                    '--total-token', str(total_token),
                    '--depth', str(depth),
                    '--top-k', str(top_k),
                    '--temperature', str(temperature),
                    '--top-k-basemodel', str(top_k_basemodel),
                    '--top-p-basemodel', str(top_p_basemodel),
                    ])
    from analyze_jsonl import cal_acc
    from analyze_jsonl import cal_time
    acc = cal_acc(savefile)
    draft_forward_times, target_forward_times, wall_times = cal_time(savefile)

    return acc, draft_forward_times, target_forward_times, wall_times

# キーの表示名を短縮するための辞書
key_alias = {
    'group_size': 'gs',
    'qdata_type': '',      # 空文字にすれば値だけ表示される (例: int4)
    'total_token': 'total',
    'depth': 'depth',
    'top_k': 'k',
    'temperature': 'temp',
    # 定義されていないキーはそのまま表示されます
}

def save_plot(filestr, png_name, labels, values, y_label, title, fmt='%.2f'):  
    plt.figure()
    bars=plt.bar(labels, values)
    plt.gca().bar_label(bars, fmt=fmt)
    plt.xticks(rotation=90)
    plt.xlabel('Parameters')
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'{filestr}_{png_name}', bbox_inches='tight')
    plt.show()

def plot_acc_vs_groupsize(filestr, png_name, configs):
    avg_acc = []
    avg_draft_times = []
    avg_target_times = []
    avg_wall_times = []
    labels = []
    for config in configs:
        acc, draft_forward_times, target_forward_times, wall_times = get_accdata(filestr, **config)
        avg_acc.append(sum(acc)/len(acc))
        avg_draft_times.append(sum(draft_forward_times)/len(draft_forward_times))
        avg_target_times.append(sum(target_forward_times)/len(target_forward_times))
        avg_wall_times.append(sum(wall_times)/len(wall_times))
        # 1. configにあるキー(k)と値(v)だけを取り出す
        # 2. key_aliasにあればその名前を、なければ元のキー名を使う
        label_parts = [f"{key_alias.get(k, k)}{v}" for k, v in config.items()]
        labels.append("_".join(label_parts))
    save_plot(filestr, f"acc_{png_name}", labels, avg_acc, 'Accuracy', 'Accuracy vs. Parameters', fmt='%.2f')
    save_plot(filestr, f"draft_time_{png_name}", labels, avg_draft_times, 'Average Draft Forward Time\n(sec / decoding steps)', 'Draft Forward Time vs. Parameters', fmt='%.4f')
    save_plot(filestr, f"target_time_{png_name}", labels, avg_target_times, 'Average Target Forward Time\n(sec /s decoding steps)', 'Target Forward Time vs. Parameters', fmt='%.4f')
    save_plot(filestr, f"wall_time_{png_name}", labels, avg_wall_times, 'Average Wall Time\n(sec / decoding steps)', 'Wall Time vs. Parameters', fmt='%.4f')


if __name__ == "__main__":
    filestr = 'data/vicuna-13b/'
    png_name = 'depthvary_token_20.png'
    configs=[
        {'qdata_type': 'W8A8','total_token':20,'depth':5},
        {'qdata_type': 'W8A8','total_token':20,'depth':10},
        {'qdata_type': 'W8A8','total_token':20,'depth':20},
        {'group_size': 32, 'qdata_type': 'int4','total_token':20,'depth':5},
        {'group_size': 32, 'qdata_type': 'int4','total_token':20,'depth':10},
        {'group_size': 32, 'qdata_type': 'int4','total_token':20,'depth':20},
        {'group_size': 32, 'qdata_type': 'int8','total_token':20,'depth':5},
        {'group_size': 32, 'qdata_type': 'int8','total_token':20,'depth':10},
        {'group_size': 32, 'qdata_type': 'int8','total_token':20,'depth':20},
        {'group_size': None, 'qdata_type': 'original','total_token':20,'depth':5},
        {'group_size': None, 'qdata_type': 'original','total_token':20,'depth':10},
        {'group_size': None, 'qdata_type': 'original','total_token':20,'depth':20},
    ]
    plot_acc_vs_groupsize(filestr,png_name, configs)



