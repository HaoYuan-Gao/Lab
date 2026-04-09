import os
from itertools import product

if __name__ == "__main__":
    modules = ["neck_seqs1", "dec_seqs1", "dec_seqs0"] # ["enc_seqs_0", "enc_seqs_1", "neck_seqs0", "neck_seqs1", "dec_seqs1", "dec_seqs0"]
    grads = ["x_grad", "h_grad", "grad_dh", "grad_wi", "grad_wh", "grad_bi", "grad_bh", "h"]
    params = [ "bias_hh_l0", "bias_ih_l0", "weight_hh_l0", "weight_ih_l0" ]

    RANKS = [f"rank{i}" for i in range(8)]
    parms_files = [f"{p}_{r}.pt" for p, r in product(params, RANKS)]
    grad_files = [f"{g}_{r}.pt" for g, r in product(grads, RANKS)]

    valid_files = set(parms_files + grad_files)   # 用 set 加速判断

    for m in modules:
        ROOT_DIR = f"gru_grad/{m}/epoch_286"
        print(f"======= process:{ROOT_DIR} =======")

        for step in os.listdir(ROOT_DIR):
            path = os.path.join(ROOT_DIR, step)

            files = os.listdir(path)
            files = set(files)

            missing = valid_files - files

            if missing:
                print(f"{m}/{step} 缺失文件 ({len(missing)} 个):")