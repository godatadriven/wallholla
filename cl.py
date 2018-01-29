import fire
from walholla.runparser import Run, load_config


def experiment(yaml_file):

    config = load_config(yaml_file)
    run = Run(config)
    df = run.execute()

    # TODO: add timestamp or something unique to filename, during development I actually like to overwrite
    # TODO: eventually disable and replace by GBQ insertion
    df.to_csv("result.csv")


if __name__ == "__main__":
    fire.Fire(experiment)
