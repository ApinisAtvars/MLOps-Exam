from azure.ai.ml import MLClient, Input, Output, command, dsl
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential

def main():
    # Connect to the workspace
    # Assuming the user has configured their config.json or we use default subscription details
    # If config.json is not present, we might need subscription_id, resource_group, workspace_name
    # For now, we'll try to load from config or use interactive login which might pick up defaults if logged in via CLI
    
    # Try to get the handle from the config file if it exists, otherwise use arguments (which we don't have here easily without hardcoding)
    # We will assume the user runs this in an environment where they are logged in or have a config.json
    try:
        credential = DefaultAzureCredential()
        # Replace these with your actual details if config.json is not found
        # ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        ml_client = MLClient.from_config(credential=credential)
    except Exception as e:
        print(f"Could not connect to workspace: {e}")
        print("Please ensure you have a config.json file or update the script with your workspace details.")
        return

    # Define the environment from conda.yaml
    env = Environment(
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file="environments/conda.yaml",
        name="got-prepare-env",
        description="Environment for data preparation"
    )

    # Define the command component
    prepare_data_component = command(
        name="prepare_data",
        display_name="Prepare Data",
        description="Preprocess data for training",
        inputs={
            "input_data": Input(type="uri_folder"), # or uri_file if it's a single file, but script uses folder
            "target_col": Input(type="string", default="house_affiliation"),
            "delimiter": Input(type="string", default=","),
            "test_size": Input(type="number", default=0.2),
            "seed": Input(type="integer", default=42),
            "stratify": Input(type="integer", default=1),
        },
        outputs={
            "train_data": Output(type="uri_folder"),
            "test_data": Output(type="uri_folder"),
        },
        # The code is in the current directory
        code="./src",
        command="""python prepare_component.py \
            --input_folder ${{inputs.input_data}} \
            --target_col ${{inputs.target_col}} \
            --delimiter ${{inputs.delimiter}} \
            --test_size ${{inputs.test_size}} \
            --seed ${{inputs.seed}} \
            --stratify ${{inputs.stratify}} \
            --out_train ${{outputs.train_data}} \
            --out_test ${{outputs.test_data}}
        """,
        environment=env,
    )

    # Define the pipeline
    @dsl.pipeline(
        compute="serverless", # Or specify a compute cluster name if you have one
        description="Pipeline to prepare GOT data"
    )
    def prepare_pipeline(input_data):
        prepare_step = prepare_data_component(
            input_data=input_data,
            target_col="house_affiliation",
            delimiter=";",
            test_size=0.2,
            seed=42,
            stratify=1
        )
        
        return {
            "train_data": prepare_step.outputs.train_data,
            "test_data": prepare_step.outputs.test_data
        }

    # Create the pipeline job
    # We need to specify the input data. 
    # We can use the registered data asset 'mlops-exam-mltable'
    # Or we can point to a local path if we want to upload it on the fly.
    
    # Using the registered data asset (version 1 or latest)
    data_input = Input(type="mltable", path="azureml:mlops-exam-mltable@latest")
    # Note: If the asset is uri_folder, use type="uri_folder". The yaml says type: mltable.
    # If the script expects a folder containing csvs, mltable might behave differently depending on how it's mounted.
    # The script uses glob(os.path.join(args.input_folder, "*.csv")).
    # If we pass an MLTable, Azure ML mounts the folder containing the MLTable file? 
    # Or does it mount the data the MLTable points to?
    # Usually for MLTable input, the script should read the MLTable. 
    # But prepare_component.py reads CSVs directly.
    # If the input is an MLTable asset, the mount might just be the folder containing the MLTable file.
    # Let's check mltable_asset.yaml again. It points to ./data.
    # If we register it, the asset will contain the files in ./data.
    # So the input_folder will contain the CSVs.
    
    pipeline_job = prepare_pipeline(input_data=data_input)
    
    # Submit the pipeline
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job,
        experiment_name="got-prepare-pipeline"
    )
    
    print(f"Pipeline submitted. Job URL: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
