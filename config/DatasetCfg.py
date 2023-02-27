from dataclasses import dataclass


# Contains all parameters for scene variability estimation dataset
@dataclass
class DatasetCfg:
    # Directory containing all raw data as per the specified file structure
    root: str = "./data"
    load: bool = True
    pca_enable: bool = True
    pca_num_features: int = 120
    splits_path: str = "./results/training_splits.pickle"
    results_path: str = "./results"

    # Parameters for attribute embedding
    class AttributeParams:
        global_loc: bool = False

    attributes: AttributeParams = AttributeParams()

    # Parameters for object embedding
    class ObjectParams:
        taxonomy: str = "rio"   # rio, eigen13, rio27, nyu40
        n_obj: int = 528        # 528, 14, 28, 41
        enabled: bool = True

    objects: ObjectParams = ObjectParams()

    # Parameters for edge embeddings
    class RelationshipParams:
        relative_loc: bool = True
        loc_threshold: float = 0.75

    relationships: RelationshipParams = RelationshipParams()

    # Parameters for variability embedding
    class VariabilityParams:
        threshold: float = 0.1

    variability: VariabilityParams = VariabilityParams()

    class TaskSimParams:
        split_path: str = "./results/task_sim_splits.pickle"
        model_path: str = "./pretrained/deltavsg_baseline.pt"

    task_sim: TaskSimParams = TaskSimParams()

    class InferenceParams:
        split_path: str = "./results/task_sim_splits.pickle"
        model_path: str = "./pretrained/deltavsg_baseline.pt"
        scans = ["20c993a5-698f-29c5-8565-40e064af0fc4",
                 "74ef8470-9dce-2d66-8339-4b51b8406cef",
                 "210cdbab-9e8d-2832-85fa-87d12badb00e",
                 "752cc583-920c-26f5-8fcc-02c767693c60",
                 "752cc585-920c-26f5-8e40-9e37e31cc861",
                 "5630cfda-12bf-2860-8511-9baf30eec4ad",
                 "0958224c-e2c2-2de1-948b-4417ac5f2711",
                 "b05fdd58-fca0-2d4f-8bd7-e80abe1ddb4c",
                 "c92fb57e-f771-2064-86e4-5f4c7c77a8c7",
                 "ddc737b3-765b-241a-9c35-6b7662c04fc9"]

    inference: InferenceParams = InferenceParams()