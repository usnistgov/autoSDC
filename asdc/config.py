import os
import confuse

PROJECTNAME = 'asdc'

class OverrideConfig(confuse.LazyConfig):
    def __init__(self, path):
        self._dynamic_config_path = path
        super().__init__(PROJECTNAME, __name__)

    def config_dir(self):
        return self._dynamic_config_path

    def read(self, user=True, defaults=True):
        """ update default config with user configuration file """
        user_config, default_config = {}, {}

        filename = self.user_config_path()
        if os.path.isfile(filename):
            user_config = confuse.load_yaml(filename)

        if self.modname:
            if self._package_path:
                filename = os.path.join(self._package_path, confuse.DEFAULT_FILENAME)
                if os.path.isfile(filename):
                    default_config = confuse.load_yaml(filename)

        default_config.update(user_config)
        config = default_config
        self.add(confuse.ConfigSource(config, 'merged', True))
