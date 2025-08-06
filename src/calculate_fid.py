import os
import argparse
import uuid
from torch_fidelity import calculate_metrics

def compute_fid(input1, input2):
    metrics = calculate_metrics(
        input1=input1,
        input2=input2,
        cuda=True,
        fid=True,
        isc=False,
        kid=False,
        verbose=False,
        input1_cache_name=f"real_{uuid.uuid4()}",
        input2_cache_name=f"gen_{uuid.uuid4()}",
    )
    return metrics['frechet_inception_distance']

def main():
    parser = argparse.ArgumentParser(description="Calculate mean FID over multiple folder pairs.")
    parser.add_argument('--real_dirs', nargs='+', required=True, help='List of real image folders')
    parser.add_argument('--gen_dirs', nargs='+', required=True, help='List of generated image folders')
    parser.add_argument('--output', default="fid_results.txt", help='Path to save results')

    args = parser.parse_args()

    if len(args.real_dirs) != len(args.gen_dirs):
        raise ValueError("Number of real_dirs and gen_dirs must match.")

    total_fid = 0
    fid_scores = []

    with open(args.output, 'a') as f:
        for real_dir, gen_dir in zip(args.real_dirs, args.gen_dirs):
            fid = compute_fid(real_dir, gen_dir)
            fid_scores.append(fid)
            total_fid += fid
            msg = f"FID between {real_dir} and {gen_dir}: {fid:.4f}"
            print(msg)
            f.write(msg + '\n')

        mean_fid = total_fid / len(fid_scores)
        summary = f"\nMean FID over {len(fid_scores)} folder pairs: {mean_fid:.4f}\n"
        print(summary)
        f.write(summary)

if __name__ == "__main__":
    main()
