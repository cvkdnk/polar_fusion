# Semantic Segmentation for Point Cloud Project

## 使用说明

### 生成配置文件

运行 `python main.py -d` 即可通过交互式命令行生成详细配置文件。

此时，所有参数均有默认数值，可以根据自己需求修改相应的参数值。 如果需要进一步了解各参数的含义，请找到参数所在文件查找函数的docstring获取更多信息。

### Builder类

修改好详细配置文件后，使用`./utils/builders.py`文件中的`Builder`类来生成一个包含所有训练所需对象的builder对象。具体来说，该对象包含如下成员：

`self.config`: 通过pyyaml读取的配置文件字典对象，包含所有的参数

`self.kitti_yaml`: 由semantic_kitti_api提供的SemanticKITTI数据集所需的参数

`self.train_loader`, `self.val_loader`, `self.test_loader`: 生成的DataLoader对象

`self.model`: 生成的模型对象

`self.loss`: 生成的损失函数

`self.loss_weight`: 损失函数权重

`self.optimizer`: 以`self.model`参数生成的优化器

### 数据读取

在生成的`self.xx_loader`对象中，可以由生成器生成一个batch的数据。在本项目中，每个batch的数据由字典组成。不经过任何数据预处理生成的batch字典数据中，`batch["Point"]`中存储了原始点云数据、`batch["Label"]`中存储了原始点云对应的语义标签数据，`batch["PointsNum"]`中存储了点云数量信息，`batch["SeqFrame"]`中存储了点云对应的文件序号和文件名。

注意，如果`batch_size`不为1时，`batch["Point"]`中数据的维度不包含batch，长度为该batch中所有点云场景的点数量之和，此时，前`batch["PointsNum"][0]`个数据为第一个场景的点云数据，以此类推。

如果在数据预处理中加入了将点云投影到其他空间中的操作，那么可以在`batch["SpaceName]`中访问相应的数据。具体请参考数据预处理中相应类的注释。

## 自定义操作

### 添加数据集支持

在`./dataloader/dataloader.py`文件中包含了现有的数据集支持类。如果现有数据集类不能满足您的需求，可以编写新的数据集类，要求包括：继承`BaseDataset`类、重写`__getitem__`、`__len__`、`gen_config_template`方法以及使用装饰器`@DatasetInerface.register`装饰该类。

`gen_config_template`方法是为了生成详细配置文件时能够直接生成默认的参数值，如果您不需要使用生成默认参数文件的功能，可以将函数体写为pass；如果需要，请参考其他类的写法，返回一个包含所有参数的字典对象。

### 添加数据预处理

数据预处理在`./dataloader/data_pipeline.py`文件中。编写新的数据预处理类的要求与编写新数据集的要求类似，需要继承`DataPipelineBaseClass`类并重写`gen_config_template`、`__call__`方法，此外，还需要改写类成员RETURN_TYPE，即数据预处理返回的数据类型，如Point、Voxel、Range以及Bev。RETURN_TYPE的作用是帮助`process_config.py`文件中从`base.yaml`生成详细配置文件的过程完成一项检查工作，确保您模型所需数据类型存在。最后，别忘了使用`DataPipelineInterface.register`装饰新的数据预处理类。

### 添加新的模型

模型文件存储在`./model/`目录下，您可以将您编写的模型文件存储在子目录中，并最终将模型的接口类写入`./model/model.py`文件中，便于`Builder`类能够识别到新加入的模型。

编写新的模型接口类与上述添加新自定义类的要求类似，新的接口类需要继承`ModuleBaseClass`类并重写`gen_config_template`方法，以及与模型类一样，重写`forward`方法。最后请用`@ModelInterface.register`装饰该模型接口类。


## 测试

在项目根目录开启Python控制台，通过`from test_interface import get_builder` 以及 `builder = get_builder(work_dir)`来获取一个builder类，其中work_dir为配置文件所在目录。然后，通过builder类的成员函数来获取模型、数据、损失函数等对象。例如，通过`builder.get_model()`来获取模型对象，通过`builder.get_dataloader()`来获取数据加载器对象。 