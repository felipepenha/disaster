from os import path

import disaster

base_path = path.dirname(path.dirname(disaster.__file__))
workspace_path = path.join(base_path, 'workspace')
data_path = path.join(workspace_path, 'data')
download_path = path.join(workspace_path, 'download')
features_path = path.join(workspace_path, 'features')
models_path = path.join(workspace_path, 'models')
predict_path = path.join(workspace_path, 'predict')
