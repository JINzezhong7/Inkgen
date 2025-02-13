
# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '8f60fba2-9902-461e-ab9e-45faf8c61df2'
resource_group = 'IA-MLOps'
workspace_name = 'ia-mlops'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_id(workspace, 'a3831e3f-a262-4e87-8fe4-bd9d45eea260')
dataset.download(target_path=r'C:\Users\v-zezhongjin\Desktop\MSRA_intern\small_set_transformer', overwrite=False)