from dataloader import DatasetInterface, DataPipelineInterface

dataset = DatasetInterface.gen_default("SemanticKITTI")['train']
data = dataset[0]

cylize = DataPipelineInterface.gen_default("Cylindrical")
rangeproj = DataPipelineInterface.gen_default("RangeProject")
data.update(cylize(data))
data.update(rangeproj(data))


