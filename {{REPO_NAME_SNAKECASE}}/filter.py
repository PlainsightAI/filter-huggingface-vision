import logging
import os
from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

os.environ['HF_HOME'] = f'{os.getcwd()}/models/hfcache'
os.environ['TRANSFORMERS_CACHE'] = f'{os.getcwd()}/models/hfcache'

__all__ = ['{{REPO_NAME_PASCALCASE}}Config', '{{REPO_NAME_PASCALCASE}}']

logger = logging.getLogger(__name__)


class {{REPO_NAME_PASCALCASE}}Config(FilterConfig):
    pass


class {{REPO_NAME_PASCALCASE}}(Filter):
    """Put help documentation here."""

    @classmethod
    def normalize_config(cls, config: {{REPO_NAME_PASCALCASE}}Config):
        config = {{REPO_NAME_PASCALCASE}}Config(super().normalize_config(config))

        # TODO: normalize and validate parameters, don't touch touch stateful resources here

        return config

    def setup(self, config: {{REPO_NAME_PASCALCASE}}Config):
        pass  # TODO: setup and connect to resources (files, databases, doomsday machines, etc...)

    def shutdown(self):
        pass  # TODO: shutdown

    def process(self, frames: dict[str, Frame]):

        # TODO: process

        return frames
    

if __name__ == '__main__':
    {{REPO_NAME_PASCALCASE}}.run()
