# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

# ml-1m -model: PSR -us: false -patch_len: 4 -stride: 2 -n_layers: 4 -n_heads: 8 -gpu_id: 0 -train_batch_size: 512 -learning_rate: 0.0001 - patch_fusion: last -patch_lamda: 1 -padding_patch: repeat
import argparse

from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 1783808(ours) < 1784320
    parser.add_argument("--model", "-m", type=str, default="SASRec", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon_Beatuy", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default='configs/MuPP.yaml', help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(    
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )
