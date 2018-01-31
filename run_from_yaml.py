import fire
from walholla.runparser import load_config
from walholla.runparser import Run


def execute(filename):
    config = load_config(filename)
    run = Run(config)
    run.execute()


if __name__=="__main__":
    fire.Fire(execute)