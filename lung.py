import pandas as pd
from radio import CTImagesMaskedBatch
from radio.dataset import FilesIndex, Dataset, Pipeline, F
from radio.models import DilatedNoduleNet
from radio.models.tf.losses import tversky_loss

nodules_df = pd.read_csv('/path/to/annotations.csv')
luna_index = FilesIndex(path='/path/to/LunaDataset/*.mhd', no_ext=True)
luna_dataset = Dataset(index=luna_index, batch_class=CTImagesMaskedBatch)

preprocessing = (Pipeline()
                 .load(fmt='raw')
                 .unify_spacing(shape=(384, 512, 512), spacing=(3.5, 2.0, 2.0)))
                 .fetch_nodules_info(nodules_df)
                 .create_mask()
                 .normalize_hu())

spacing_randomizer = lambda *args: 0.2 * np.random.uniform(size=3) + [3.5, 2.0, 2.0]
augmentation = (Pipeline()                 
                .sample_nodules(nodule_size=(48, 76, 76))
                .rotate(random=True, angle=30, mask=True)
                .unify_spacing(spacing=F(spacing_randomizer), shape=(32, 64, 64)))


vnet_config = {'loss': tversky_loss,
               'inputs': dict(images={'shape': (32, 64, 64, 1)},
                              labels={'name': 'targets', 'shape': (32, 64, 64, 1)})}
vnet_config['input_block/inputs'] = 'images'
model_training = (Pipeline()
                  .init_model(name='vnet', model_class=DilatedNoduleNet, config=vnet_config)
                  .train_model(name='vnet', feed_dict={'images': F(CTIMB.unpack, component='images'),
                                                       'targets': F(CTIMB.unpack, component='masks')})
                
workflow = (preprocessing + augmentation + model_training)
(luna_dataset >> workflow).run(batch_size=4)
