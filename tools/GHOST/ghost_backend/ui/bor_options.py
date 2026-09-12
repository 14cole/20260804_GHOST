"""Desktop controls for BOR execution, separate from 2D profiles."""
from PySide6.QtWidgets import QGroupBox, QFormLayout, QComboBox, QSpinBox
from ghost_backend.bor.options import validate_options


class BorOptionsWidget(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('BOR execution and resources', parent)
        form=QFormLayout(self)
        self.factorization=QComboBox()
        self.factorization.addItem('Dense LU', 'dense')
        self.factorization.addItem('Compressed assembly (experimental)', 'compressed')
        form.addRow('Factorization',self.factorization)
        self.batch=QSpinBox()
        self.batch.setRange(1,256)
        self.batch.setValue(64)
        form.addRow('Aspects per batch',self.batch)
        self.reuse=QComboBox()
        for label,value in [('Automatic','auto'),('Off','off'),('On','on')]:
            self.reuse.addItem(label,value)
        form.addRow('Incident basis reuse',self.reuse)
        self.storage=QSpinBox()
        self.storage.setRange(16,1048576)
        self.storage.setValue(2048)
        self.storage.setSuffix(' MiB')
        self.storage.setToolTip('Combined numeric operator and inverse storage across active modes. Workspaces and near quadrature require additional RAM.')
        form.addRow('Compressed storage cap',self.storage)
        self.tile=QSpinBox()
        self.tile.setRange(8,128)
        self.tile.setValue(32)
        form.addRow('Compression tile size',self.tile)
        self.cache=QSpinBox()
        self.cache.setRange(0,4096)
        self.cache.setValue(16)
        self.cache.setSuffix(' MiB')
        self.cache.setToolTip('Shared cache for repeated compressed coefficient queries; 0 disables it. Additional to the compressed storage cap.')
        form.addRow('Coefficient tile cache',self.cache)
        self.storage.setEnabled(False)
        self.tile.setEnabled(False)
        self.cache.setEnabled(False)
        self.factorization.currentIndexChanged.connect(self._sync)

    def _sync(self):
        compressed=self.factorization.currentData()=='compressed'
        self.storage.setEnabled(compressed)
        self.tile.setEnabled(compressed)
        self.cache.setEnabled(compressed)

    def set_value(self, raw):
        value=validate_options(raw)
        self.factorization.setCurrentIndex(self.factorization.findData(value['factorization']))
        self.batch.setValue(value['angle_batch_size'])
        self.reuse.setCurrentIndex(self.reuse.findData(value['rhs_compression']))
        self.storage.setValue(value['compressed_storage_mib'])
        self.tile.setValue(value['compression_tile'])
        self.cache.setValue(value['tile_cache_mib'])
        self._sync()

    def value(self):
        return validate_options(dict(factorization=self.factorization.currentData(),
            angle_batch_size=self.batch.value(),rhs_compression=self.reuse.currentData(),
            compressed_storage_mib=self.storage.value(),compression_tile=self.tile.value(),
            tile_cache_mib=self.cache.value()))
