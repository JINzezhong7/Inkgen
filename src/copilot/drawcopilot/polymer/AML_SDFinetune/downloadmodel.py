
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset
import sys
import os
def downloadmodel(path):
    subscription_id = '8f60fba2-9902-461e-ab9e-45faf8c61df2'
    resource_group = 'IA-MLOps'
    workspace_name = 'ia-mlops'

    workspace = Workspace(subscription_id, resource_group, workspace_name)

    dataset = Dataset.get_by_id(workspace, path)
    if not os.path.exists('downloads'):
        os.makedirs('downloads')
    dataset.download(target_path='downloads', overwrite=True)

if __name__ == '__main__':
    downloadmodel(sys.argv[1])