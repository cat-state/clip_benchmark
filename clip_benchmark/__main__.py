from fire import Fire

from . import clip_benchmark, plot_models

if __name__ == "__main__":
    Fire(dict(benchmark=clip_benchmark, plots=plot_models))
