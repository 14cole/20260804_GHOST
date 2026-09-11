"""Reusable report definitions and explicit transfer from the Plotting canvas."""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile
from xml.etree import ElementTree as ET

COMBOS = ('plot_type_combo', 'frequency_azimuth_mode_combo', 'elevation_combo', 'azimuth_combo',
          'polarization_combo', 'x_scale_mode_combo', 'scale_mode_combo', 'legend_mode_combo', 'global_line_style_combo')
SPINS = ('azimuth_band_min_spin', 'azimuth_band_max_spin', 'azimuth_percentile_spin',
         'x_min_spin', 'x_max_spin', 'x_step_spin', 'y_min_spin', 'y_max_spin', 'y_step_spin', 'global_line_width_spin')
TEXTS = ('deck_title_edit', 'template_edit', 'azimuth_layout_edit', 'frequency_layout_edit')


def write_report_recipe(path, state):
    path = Path(path)
    if not path.name.endswith('.report.json'):
        raise ValueError('Use the .report.json suffix for report recipes.')
    data = json.dumps({'schema':'grim.report-recipe','version':1,'setup':state}, indent=2, allow_nan=False)
    fd,temporary = tempfile.mkstemp(dir=path.parent,prefix='.report-recipe-')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream: stream.write(data+'\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def read_report_recipe(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value,dict) or value.get('schema')!='grim.report-recipe' or value.get('version')!=1:
        raise ValueError('Choose a supported GRIM report recipe (version 1).')
    return value['setup']


def inspect_template(path, layouts):
    """Read template shape and named layouts without editing a presentation."""
    layouts = [str(x).strip() for x in layouts if str(x).strip()]
    if not path:
        if layouts: raise ValueError('Named layouts require a PowerPoint template. Select one or clear the layout names.')
        return 'Generic widescreen presentation.'
    path = Path(path)
    if path.suffix.lower() not in ('.pptx','.potx') or not path.is_file():
        raise ValueError(f'Choose an existing .pptx or .potx template: {path}')
    ns = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        presentation = ET.fromstring(archive.read('ppt/presentation.xml'))
        size = presentation.find('p:sldSz',ns)
        if size is None or not math.isclose(int(size.attrib['cx'])/int(size.attrib['cy']),16/9,rel_tol=.005):
            raise ValueError('Template must use widescreen 16:9 slides.')
        names=[]
        for member in archive.namelist():
            if member.startswith('ppt/slideLayouts/') and member.endswith('.xml') and '/_rels/' not in member:
                root = ET.fromstring(archive.read(member))
                slide = root.find('p:cSld',ns)
                if slide is not None: names.append(slide.attrib.get('name',''))
        for selector in layouts:
            name = selector.split('::')[-1].strip()
            matches = [n for n in names if n.casefold()==name.casefold()]
            if not matches: raise ValueError(f'Layout {selector!r} is unavailable. Available: '+', '.join(names))
            if len(matches)>1 and '::' not in selector:
                raise ValueError(f'Layout {name!r} is ambiguous. Qualify it as Master :: Layout.')
    return 'Template is widescreen; requested layout names found.'


def _plain(value):
    return value.item() if hasattr(value,'item') else value


def current_plot_report_setup(window):
    """Transfer the last successful Plotting view, using its frozen input units."""
    from GRIM_Backend.reports.plot_data import NamedGrid, get_plot_availability
    from GRIM_Backend.plotting.modes.common import axis_unit, convert_axis_values
    context = window._plot_contexts['plotting']
    # The active context is bound to the host; inactive contexts store the snapshot.
    spec = window.last_python_plot_spec if window._active_plot_tab == 'plotting' else context.last_python_plot_spec
    if not spec or spec[0]!='supported' or spec[3] not in ('azimuth_rect','azimuth_polar','frequency','elevation_sweep'):
        raise ValueError('Draw a rectangular/polar azimuth, elevation, or frequency plot first. Hold, PBP, and image plots need their own export workflow.')
    params=spec[4]
    if params.get('phase') or params.get('scale')=='linear':
        raise ValueError('The report supports magnitude in dB. Switch the Plotting view to dB magnitude before copying it.')
    if spec[3]=='azimuth_polar' and params.get('polar_zero','N')!='N':
        raise ValueError('PPT polar reports use North as zero. Set Plotting polar zero to North before copying.')
    selected=[ref.dataset_id for ref in spec[1]]
    catalog=window.ppt_workspace._catalog
    if any(key not in catalog for key in selected):
        raise ValueError('A plotted dataset was removed. Redraw the plot with the current datasets.')
    reference=catalog[selected[int(params.get('reference_index',0))]].grid
    availability=get_plot_availability([NamedGrid(catalog[key].name,catalog[key].grid) for key in selected], evaluate_phase=False)
    state=window.ppt_workspace.capture_report_setup()
    state['selected']=selected
    state['order']=selected+[k for k in state['order'] if k not in selected]
    state['units']={key:getattr(availability,key) for key in ('azimuth_unit','elevation_unit','frequency_unit','rcs_unit','angular_coordinate_system')}
    mode='elevation' if spec[3]=='elevation_sweep' else spec[3]
    state['combos']['plot_type_combo']=mode
    state['combos']['frequency_azimuth_mode_combo']='exact'
    state['combos']['polarization_combo']=params['polarization']
    for axis in ('azimuth','elevation'):
        values=params[axis+'s']
        fixed = axis=='azimuth' and mode in ('frequency','elevation') or axis=='elevation' and mode!='elevation'
        if fixed and len(values)!=1:
            raise ValueError(f'This report requires one {axis} cut. Select one in Plotting and redraw.')
        state['combos'][axis+'_combo'] = (float(convert_axis_values([values[0]],axis,axis_unit(reference,axis),getattr(availability,axis+'_unit'))[0]) if values else None)
    state['frequencies']=[float(v) for v in convert_axis_values(params['frequencies'],'frequency',axis_unit(reference,'frequency'),availability.frequency_unit)]
    if mode=='frequency' and set(state['frequencies'])!=set(availability.frequencies):
        raise ValueError('Frequency reports sweep the full dataset axis. Select all frequencies in Plotting and redraw before copying, or configure the report directly.')
    axes = window.plot_ax if window._active_plot_tab=='plotting' else context.plot_ax
    axis = 'frequency' if mode=='frequency' else 'elevation' if mode=='elevation' else 'azimuth'
    limits=axes.get_xlim()
    if mode=='azimuth_polar':
        xlimits=[-180.,180.]
    else:
        xlimits=convert_axis_values(limits,axis,axis_unit(reference,axis),'GHz' if axis=='frequency' else 'deg')
    ylimits=axes.get_ylim()
    state['combos']['x_scale_mode_combo']='fixed'
    state['combos']['scale_mode_combo']='fixed'
    state['combos']['legend_mode_combo']='per_plot' if params.get('show_legend',True) else 'none'
    for direction,limits in [('x',xlimits),('y',ylimits)]:
        state['spins'][direction+'_min_spin']=float(limits[0])
        state['spins'][direction+'_max_spin']=float(limits[1])
        ticks=axes.get_xticks() if direction=='x' else axes.get_yticks()
        if direction=='x' and mode!='azimuth_polar':
            ticks=convert_axis_values(ticks,axis,axis_unit(reference,axis),'GHz' if axis=='frequency' else 'deg')
        if direction=='x' and mode=='azimuth_polar': ticks=[0.,45.]
        state['spins'][direction+'_step_spin']=float(abs(ticks[1]-ticks[0])) if len(ticks)>1 else float(abs(limits[1]-limits[0])/5)
    from matplotlib.colors import to_hex
    widths,styles,colors={},{},{}
    for line in axes.lines:
        key=getattr(line,'_grim_dataset_key',None)
        if key in selected:
            if line.get_linestyle() not in ('-','--',':','-.'):
                raise ValueError('A plotted series uses markers without a supported line style. Use a line plot before copying.')
            width,style=float(line.get_linewidth()),line.get_linestyle()
            color=to_hex(line.get_color())
            if key in widths and (widths[key]!=width or styles[key]!=style or colors[key]!=color):
                raise ValueError('One dataset has different line styles across cuts. Set a consistent dataset style before copying.')
            widths[key],styles[key]=width,style
            colors[key]=color
    state['line_widths'],state['line_styles']=widths,styles
    state['line_colors']=colors
    return state


class ReportWorkflowMixin:
    def _build_report_workflow(self, layout):
        from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox
        group = QGroupBox('Reusable report setup')
        body = QVBoxLayout(group)
        row = QHBoxLayout()
        self.save_report_recipe_button = QPushButton('Save recipe…')
        self.load_report_recipe_button = QPushButton('Load recipe…')
        row.addWidget(self.save_report_recipe_button)
        row.addWidget(self.load_report_recipe_button)
        body.addLayout(row)
        self.recipe_current_datasets = QCheckBox('Use current PPT datasets when loading')
        self.recipe_current_datasets.setToolTip('Apply the saved report definition to the current selection. Missing cuts and channels are reported before the setup is changed.')
        body.addWidget(self.recipe_current_datasets)
        self.use_current_plot_button = QPushButton('Use current Plotting setup')
        self.use_current_plot_button.setEnabled(self._current_plot_provider is not None)
        body.addWidget(self.use_current_plot_button)
        self.check_presentation_button = QPushButton('Check PowerPoint and template')
        body.addWidget(self.check_presentation_button)
        self.open_presentation_button = QPushButton('Open exported presentation')
        self.open_presentation_button.setEnabled(False)
        body.addWidget(self.open_presentation_button)
        self._last_exported_presentation = None
        self.save_report_recipe_button.clicked.connect(self._save_report_recipe)
        self.load_report_recipe_button.clicked.connect(self._load_report_recipe)
        self.use_current_plot_button.clicked.connect(self._use_current_plot_setup)
        self.check_presentation_button.clicked.connect(self._check_presentation)
        self.open_presentation_button.clicked.connect(self._open_exported_presentation)
        layout.addWidget(group)

    def capture_report_setup(self):
        units = {}
        if self._availability:
            units = {key:getattr(self._availability,key) for key in ('azimuth_unit','elevation_unit','frequency_unit','rcs_unit','angular_coordinate_system')}
        return dict(combos={key:_plain(getattr(self,key).currentData()) for key in COMBOS},
                    spins={key:getattr(self,key).value() for key in SPINS},
                    texts={key:getattr(self,key).text() for key in TEXTS},
                    frequencies=[_plain(v) for v in self.selected_frequencies()],
                    order=list(self.dataset_ids_in_order()), selected=list(self.selected_dataset_ids()),
                    datasets={key:{'name':entry.name,'source':str(entry.source or getattr(entry.grid,'source_path','') or '')} for key,entry in self._catalog.items()},
                    line_widths=dict(self._series_line_widths), line_styles=dict(self._series_line_styles), line_colors=dict(self._series_line_colors), units=units)

    def apply_report_setup(self, raw, *, current_datasets=False):
        if self.job_is_running(): raise ValueError('Wait for the report operation to finish.')
        previous = self.capture_report_setup()
        previous_axes = copy.deepcopy(self._x_axis_settings)
        previous_customized = set(self._x_axis_customized)
        value = copy.deepcopy(raw)
        required = {'combos','spins','texts','frequencies','order','selected','datasets','line_widths','line_styles','line_colors','units'}
        if not isinstance(value,dict) or set(value)!=required:
            raise ValueError('Report recipe has missing or unsupported fields.')
        for key,expected in [('combos',COMBOS),('spins',SPINS),('texts',TEXTS)]:
            if not isinstance(value[key],dict) or set(value[key])!=set(expected):
                raise ValueError(f'Invalid report recipe {key}.')
        for key,x in value['spins'].items():
            widget=getattr(self,key)
            if isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or (key not in ('azimuth_band_min_spin','azimuth_band_max_spin') and not widget.minimum() <= x <= widget.maximum()):
                raise ValueError(f'{key.removesuffix("_spin")}: value is outside the supported range.')
        for key,x in value['texts'].items():
            if not isinstance(x,str): raise ValueError(f'{key}: text required.')
        for key in ('selected','order','frequencies'):
            if not isinstance(value[key],list) or len(value[key]) != len(set(value[key])):
                raise ValueError(f'Report {key} must be a list of unique values.')
        if not isinstance(value['datasets'],dict) or not isinstance(value['line_widths'],dict) or not isinstance(value['line_styles'],dict):
            raise ValueError('Invalid dataset references or line styles.')
        if current_datasets:
            # Styles follow the saved selected-series order when applying to a new selection.
            remap = dict(zip(value['selected'],previous['selected']))
            value['selected'],value['order'] = previous['selected'],previous['order']
        else:
            remap={}
            for key in value['order']:
                if key in self._catalog:
                    remap[key]=key
                    continue
                ref=value['datasets'].get(key,{})
                matches=[i for i,entry in self._catalog.items() if ref.get('source') and str(entry.source or getattr(entry.grid,'source_path','') or '')==ref['source']]
                if not matches:
                    matches=[i for i,entry in self._catalog.items() if entry.name==ref.get('name')]
                if len(matches)==1: remap[key]=matches[0]
                elif key in value['selected']:
                    raise ValueError(f'Dataset {ref.get("name",key)!r} is missing or ambiguous. Load it, or enable Use current PPT datasets when loading.')
            value['selected']=[remap[key] for key in value['selected']]
            value['order']=[remap[key] for key in value['order'] if key in remap]
        if len(value['selected'])!=len(set(value['selected'])):
            raise ValueError('Two recipe datasets map to one loaded dataset. Rename or choose current datasets explicitly.')
        from matplotlib.colors import is_color_like
        if not isinstance(value['line_colors'],dict) or any(not isinstance(x,str) or not is_color_like(x) for x in value['line_colors'].values()):
            raise ValueError('Invalid series color.')
        for key in ('line_widths','line_styles','line_colors'):
            value[key]={remap[k]:v for k,v in value[key].items() if k in remap}
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) or not .5<=x<=5 for x in value['line_widths'].values()):
            raise ValueError('Series line widths must be between 0.5 and 5 pt.')
        if any(x not in ('-','--',':','-.') for x in value['line_styles'].values()):
            raise ValueError('Unsupported series line style.')
        try:
            self._install_report_setup(value, strict=True)
            self._build_plan()
        except Exception:
            self._install_report_setup(previous,strict=False)
            self._x_axis_settings = previous_axes
            self._x_axis_customized = previous_customized
            raise
        self._mark_preview_stale()
        self._last_error = ''
        self._set_status('Report setup loaded. Build and review the preview before exporting.')

    def _install_report_setup(self,value,*,strict):
        from GRIM_Backend.reports.workspace import _find_combo_value, _AXIS_VALUE_ROLE, _axis_values_equal, _CATALOG_ID_ROLE
        ordered=[key for key in value['order'] if key in self._catalog]
        ordered.extend(key for key in self.dataset_ids_in_order() if key not in ordered)
        self.dataset_list.blockSignals(True)
        items={str(self.dataset_list.item(i).data(_CATALOG_ID_ROLE)): self.dataset_list.item(i) for i in range(self.dataset_list.count())}
        held=[]
        while self.dataset_list.count(): held.append(self.dataset_list.takeItem(0))
        for key in ordered: self.dataset_list.addItem(items[key])
        self.dataset_list.blockSignals(False)
        self.select_dataset_ids(value['selected'])
        combos=dict(value['combos'])
        spins=dict(value['spins'])
        if strict and value['units'] and self._availability:
            new={k:getattr(self._availability,k) for k in value['units']}
            for k in ('rcs_unit','angular_coordinate_system'):
                if new[k]!=value['units'][k]: raise ValueError(f'Recipe {k} differs from the current datasets. Use a matching report definition.')
            factors={'deg':1.,'rad':180/math.pi,'GHz':1.,'MHz':1e-3,'kHz':1e-6,'Hz':1e-9}
            for axis,field in [('elevation','elevation_combo'),('azimuth','azimuth_combo')]:
                if combos[field] is not None:
                    combos[field] *= factors[value['units'][axis+'_unit']]/factors[new[axis+'_unit']]
            ratio=factors[value['units']['frequency_unit']]/factors[new['frequency_unit']]
            frequencies=[v*ratio for v in value['frequencies']]
            ratio=factors[value['units']['azimuth_unit']]/factors[new['azimuth_unit']]
            for key in ('azimuth_band_min_spin','azimuth_band_max_spin'):
                spins[key] *= ratio
        else: frequencies=value['frequencies']
        for key in COMBOS:
            widget=getattr(self,key)
            target=combos[key]
            index=_find_combo_value(widget,target)
            kind=combos['plot_type_combo']
            inactive = ((key=='azimuth_combo' and (kind in ('azimuth_rect','azimuth_polar') or (kind=='frequency' and combos['frequency_azimuth_mode_combo']=='band')))
                        or (key=='elevation_combo' and kind=='elevation'))
            if index<0 and inactive:
                continue
            if index<0 and strict and target is not None:
                raise ValueError(f'Unavailable {key.removesuffix("_combo").replace("_"," ")}: {target}. Choose matching datasets or revise the recipe.')
            widget.setCurrentIndex(index)
        if strict and combos['plot_type_combo']!='frequency':
            available=[self.frequency_list.item(i).data(_AXIS_VALUE_ROLE) for i in range(self.frequency_list.count())]
            missing=[v for v in frequencies if not any(_axis_values_equal(v,x) for x in available)]
            if missing: raise ValueError('Unavailable report frequencies: '+', '.join(map(str,missing))+'. No substitute cuts were selected.')
        self.select_frequencies(frequencies)
        # Availability and plot-family signals can reset default axes; apply saved overrides last.
        for key,x in spins.items():
            widget=getattr(self,key)
            if strict and key.startswith('azimuth_band_') and combos['frequency_azimuth_mode_combo']=='band' and not widget.minimum() <= x <= widget.maximum():
                raise ValueError('Saved azimuth band extends outside the current dataset coverage.')
            widget.setValue(x)
        for key,x in value['texts'].items(): getattr(self,key).setText(x)
        self._series_line_widths=dict(value['line_widths'])
        self._series_line_styles=dict(value['line_styles'])
        self._series_line_colors=dict(value['line_colors'])
        self._refresh_series_style_datasets()

    def _save_report_recipe(self):
        from PySide6.QtWidgets import QFileDialog
        try:
            self._build_plan()
            path,_=QFileDialog.getSaveFileName(self,'Save report recipe','report.report.json','Report recipe (*.report.json)')
            if path:
                if not path.endswith('.report.json'): path+='.report.json'
                write_report_recipe(path,self.capture_report_setup())
                self._set_status(f'Report recipe saved: {path}')
        except Exception as exc: self._show_error(str(exc))

    def _load_report_recipe(self):
        from PySide6.QtWidgets import QFileDialog
        path,_=QFileDialog.getOpenFileName(self,'Load report recipe','','Report recipe (*.report.json)')
        if path:
            try: self.apply_report_setup(read_report_recipe(path),current_datasets=self.recipe_current_datasets.isChecked())
            except Exception as exc: self._show_error(str(exc))

    def _use_current_plot_setup(self):
        try:
            if self._current_plot_provider is None: raise ValueError('No Plotting workspace is connected.')
            self.apply_report_setup(self._current_plot_provider())
            self._set_status('Plotting cuts, axes, legend, and line styles copied. Report template and title retained. Build a preview to review the slide layout.')
        except Exception as exc: self._show_error(str(exc))

    def _check_presentation(self):
        if self.job_is_running(): return
        from PySide6.QtCore import QObject, QThread, Signal, Slot
        path=self.template_edit.text().strip()
        layouts=[self.frequency_layout_edit.text()] if self.plot_type_combo.currentData()=='frequency' else [self.azimuth_layout_edit.text()]
        class Worker(QObject):
            done=Signal(str)
            @Slot()
            def run(worker):
                try:
                    result=inspect_template(path,layouts)
                    from GRIM_Backend.reports.report import PowerPointComBridge
                    bridge=PowerPointComBridge()
                    if path: bridge.preflight_template(Path(path),layouts)
                    else: bridge.preflight()
                    worker.done.emit('PowerPoint is available. '+result)
                except Exception as exc: worker.done.emit('Correct before exporting: '+str(exc))
        self._set_busy(True)
        self._set_status('Checking PowerPoint and template…')
        self._thread=QThread(self)
        self._worker=Worker()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._set_status)
        self._worker.done.connect(self._thread.quit)
        self._worker.done.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._export_thread_finished)
        self._thread.start()

    def _open_exported_presentation(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        path=self._last_exported_presentation
        if not path or not Path(path).is_file():
            self._show_error('The exported presentation has moved or is unavailable.')
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve()))):
            self._show_error('Windows could not open the presentation. Open it from its saved folder.')
