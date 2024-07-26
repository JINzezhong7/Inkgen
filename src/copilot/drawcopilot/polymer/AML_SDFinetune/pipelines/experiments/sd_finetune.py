# pylint: disable=no-member
# NOTE: because it raises 'dict' has no 'outputs' member in dsl.pipeline construction
import os
import sys

from azure.ml.component import dsl
from shrike.pipeline import AMLPipelineHelper

# NOTE: if you need to import from pipelines.*
ACCELERATOR_ROOT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if ACCELERATOR_ROOT_PATH not in sys.path:
    print(f"Adding to path: {ACCELERATOR_ROOT_PATH}")
    sys.path.append(str(ACCELERATOR_ROOT_PATH))


class SDXL_Finetune(AMLPipelineHelper):
    """Runnable/reusable pipeline helper class

    This class inherits from AMLPipelineHelper which provides
    helper functions to create reusable production pipelines.
    """

    def build(self, config):
        """Builds a pipeline function for this pipeline using AzureML SDK (dsl.pipeline).

        This method returns a constructed pipeline function
        (decorated with @dsl.pipeline).

        Args:
            config (DictConfig): configuration object

        Returns:
            dsl.pipeline: the function to create your pipeline
        """

        # helper functions below load the subgraph/component from
        # registered or local version depending on your config.run.use_local
        sd_downloadbasemodel_component = self.component_load("sd_downloadbasemodel")
        sd_finetune_component = self.component_load("sd_finetune")
        sd_package_component = self.component_load("sd_package")
        hed_train_component = self.component_load("hednet")
        # Here you should create an instance of a pipeline function
        # (using your custom config dataclass)
        @dsl.pipeline(
            name="sd_finetune",
            description="",
            default_datastore=config.compute.compliant_datastore,
        )
        def demo_pipeline_function(image_path, heddata_path):
            """Pipeline function for this graph.

            Args:
                demo_dataset: input dataset

            Returns:
                dict[str->PipelineOutputData]: a dictionary of your pipeline outputs
                    for instance to be consumed by other graphs
            """
            # general syntax:
            # component_instance = component_class(input=data, param=value)
            # or
            # subgraph_instance = subgraph_function(input=data, param=value)
            env_variables = {"AZUREML_COMMON_RUNTIME_USE_SBOM_CAPABILITY": "True"}
            basemodeldownload_step = sd_downloadbasemodel_component()
            self.apply_recommended_runsettings(
                "sd_downloadbasemodel",
                basemodeldownload_step,
                gpu=False,
                **{"environment_variables": env_variables}
            )

            finetune_component_step = sd_finetune_component(
                pretrained_model_name_or_path = basemodeldownload_step.outputs.model_output_dir,
                train_data_dir = image_path,
                mixed_precision = config.inputs.mixed_precision,
                num_processes = config.inputs.num_processes,
                num_machines = config.inputs.num_machines,
                num_cpu_threads_per_process = config.inputs.num_cpu_threads_per_process,
                multires_noise_discount = config.inputs.multires_noise_discount,
                multires_noise_iterations = config.inputs.multires_noise_iterations,
                bucket_reso_steps = config.inputs.bucket_reso_steps,
                caption_extension = config.inputs.caption_extension,
                clip_skip = config.inputs.clip_skip,
                min_bucket_reso = config.inputs.min_bucket_reso,
                max_bucket_reso = config.inputs.max_bucket_reso,
                huber_c = config.inputs.huber_c,
                huber_schedule = config.inputs.huber_schedule,
                learning_rate = config.inputs.learning_rate,
                loss_type = config.inputs.loss_type,
                lr_scheduler = config.inputs.lr_scheduler,
                lr_scheduler_num_cycles = config.inputs.lr_scheduler_num_cycles,
                lr_warmup_steps = config.inputs.lr_warmup_steps,
                max_data_loader_n_workers = config.inputs.max_data_loader_n_workers,
                max_grad_norm = config.inputs.max_grad_norm,
                resolution = config.inputs.resolution,
                max_train_steps = config.inputs.max_train_steps,
                min_timestep = config.inputs.min_timestep,
                network_alpha = config.inputs.network_alpha,
                network_dim = config.inputs.network_dim,
                network_module = config.inputs.network_module,
                noise_offset = config.inputs.noise_offset,
                optimizer_type = config.inputs.optimizer_type,
                output_name = config.inputs.output_name,
                save_every_n_epochs = config.inputs.save_every_n_epochs,
                save_model_as = config.inputs.save_model_as,
                save_precision = config.inputs.save_precision,
                text_encoder_lr = config.inputs.text_encoder_lr,
                train_batch_size = config.inputs.train_batch_size,
                unet_lr = config.inputs.unet_lr)

            self.apply_recommended_runsettings(
                "sd_finetune",
                finetune_component_step,
                gpu=True,
                **{"environment_variables": env_variables}
            )

            hed_train_step = hed_train_component(heddatadir = heddata_path)
            self.apply_recommended_runsettings(
                "hednet",
                hed_train_step,
                gpu=True,
                **{"environment_variables": env_variables}
            )

            package_step = sd_package_component(lora_model_dir = finetune_component_step.outputs.output_dir, hednet_model_dir = hed_train_step.outputs.model_output_dir)
            self.apply_recommended_runsettings(
                "sd_package",
                package_step,
                gpu=False,
                **{"environment_variables": env_variables}
            )
        

        # finally return the function itself to be built by helper code
        return demo_pipeline_function

    def pipeline_instance(self, pipeline_function, config):
        """Given a pipeline function, creates a runnable instance based on provided config.

        This is used only when calling this as a runnable pipeline
        using .main() function (see below).
        The goal of this function is to map the config to the
        pipeline_function inputs and params.

        Args:
            pipeline_function (function):
                the pipeline function obtained from self.build()
            config (DictConfig):
                configuration object

        Returns:
            azureml.core.Pipeline: the instance constructed
                                   with its inputs and params.
        """

        # NOTE: self.dataset_load() helps to load the dataset
        # based on its name and version
        image_path = self.dataset_load(
            name=config.inputs.train_data_dir,
            version=config.inputs.train_data_dir_version,
        )

        hed_path = self.dataset_load(
            name=config.inputs.heddatadir,
            version=config.inputs.heddatadir_version,
        )
        
        # we simply call the pipeline function
        demo_pipeline = pipeline_function(image_path=image_path, heddata_path=hed_path)

        # and we return that function so that helper can run it.
        return demo_pipeline


# NOTE: main block is necessary only if script is intended to be run from command line
if __name__ == "__main__":
    # calling the helper .main() function
    SDXL_Finetune.main()
