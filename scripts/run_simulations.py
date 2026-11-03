"""Fit every (design, hypothesis, prior, replicate) combination, skipping completed fits.

Fits run one after the other, each in its own process. Use --list to write the
list of commands, e.g. to submit them as a Slurm array (scripts/slurm_array.sh).

Examples:
    python scripts/run_simulations.py                          # everything (240 fits)
    python scripts/run_simulations.py --design p134 --prior horseshoe
    python scripts/run_simulations.py --list > jobs.txt
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pgxnlme.data import DESIGNS, HYPOTHESES, fit_directory, list_datasets  # noqa: E402
from pgxnlme.priors import PRIORS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--design", nargs="+", choices=DESIGNS, default=list(DESIGNS))
    parser.add_argument("--hypothesis", nargs="+", choices=HYPOTHESES, default=list(HYPOTHESES))
    parser.add_argument("--prior", nargs="+", choices=PRIORS, default=list(PRIORS))
    parser.add_argument("--replicates", type=int, nargs="+", help="default: all available")
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--list", action="store_true", help="print the commands of the remaining fits and exit")
    args = parser.parse_args()

    commands = []
    for path in list_datasets(args.design, args.hypothesis):
        design, hypothesis, replicate = path.parts[-3], path.parts[-2], int(path.stem.split("_")[1])
        if args.replicates and replicate not in args.replicates:
            continue
        for prior in args.prior:
            if (fit_directory(args.results, design, hypothesis, prior, replicate) / "diagnostics.json").exists():
                continue
            commands.append([sys.executable, str(ROOT / "scripts" / "fit.py"), "--design", design,
                             "--hypothesis", hypothesis, "--prior", prior, "--replicate", str(replicate),
                             "--results", args.results])
    if args.list:
        print("\n".join(" ".join(command) for command in commands))
        return
    print(f"{len(commands)} fits to run")
    for i, command in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] {' '.join(command[2:])}", flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
