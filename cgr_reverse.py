import numpy as np
import pandas as pd
import sys


vertex = {
    'A': (0, 0),
    'C': (0, 1),
    'G': (1, 1),
    'U': (1, 0)
}

def cgr_encode(seq, res=64):
    mat = np.zeros((res, res), dtype=int)
    x, y = 0.5, 0.5
    for base in seq:
        if base not in vertex:
            continue
        vx, vy = vertex[base]
        x = (x + vx) / 2
        y = (y + vy) / 2
        ix = int(x * (res - 1))
        iy = int(y * (res - 1))
        mat[iy, ix] += 1
    return mat


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("用法: python3 cgr_encode.py <文件路径前缀> <编码模式> <分辨率>")
        print("示例: python3 cgr_encode.py train_out_converted rna 64")
        print("示例: python3 cgr_encode.py test_out_converted rna 64")
        sys.exit(1)


    pth = sys.argv[1]
    mode = sys.argv[2]
    res = int(sys.argv[3])


    input_csv = f'{pth}.csv'
    output_txt = f'{pth}_rev-32.txt'

    print(f"输入文件: {input_csv}")
    print(f"输出文件: {output_txt}")
    print(f"编码模式: {mode}")
    print(f"分辨率: {res}x{res}")

    try:
        df = pd.read_csv(input_csv)

        with open(output_txt, 'w') as fout:

            fout.write('sequence,label,accession\n')

            for idx, row in df.iterrows():
                seq = row['sequence'].replace('T', 'U')


                mat_fwd = cgr_encode(seq, res)
                mat_rev = cgr_encode(seq[::-1], res)


                mat_combined = np.concatenate([mat_fwd.flatten(), mat_rev.flatten()])


                cgr_vector = ' '.join(map(str, mat_combined))

                fout.write(f'{cgr_vector},{row["label"]},{row["accession"]}\n')

        print(f'CGR编码完成（包含正向与逆序），输出文件：{output_txt}')

    except FileNotFoundError:
        print(f"错误：找不到输入文件 {input_csv}")
        sys.exit(1)
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)
