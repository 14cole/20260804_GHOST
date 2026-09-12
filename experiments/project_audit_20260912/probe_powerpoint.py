"""Exercise GRIM's actual PowerPoint bridge on audit-owned synthetic data."""
import json
from pathlib import Path
import traceback
import zipfile
from xml.etree import ElementTree as ET
import probe_workflows as p

record={}
try:
    p.load()
    p.select(0,1)
    p.ppt()
    workspace=p.window.ppt_workspace
    output=p.AUDIT/'powerpoint-audit.pptx'
    workspace.output_edit.setText(str(output))
    assert not output.exists(), 'Keep previous audit export; use a fresh output path'
    assert workspace.export_report(), workspace._last_error
    p.wait(workspace.job_is_running,120)
    assert output.is_file(), workspace._last_error
    with zipfile.ZipFile(output) as z:
        slides=[name for name in z.namelist() if name.startswith('ppt/slides/slide') and name.endswith('.xml')]
        record={'output':str(output),'bytes':output.stat().st_size,'slides':len(slides),'zip_crc_error':z.testzip()}
        root=ET.fromstring(z.read('ppt/presentation.xml'))
        size=root.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldSz')
        record['size_emu']=size.attrib
    from GRIM_Backend.reports.report import PowerPointComBridge
    import pythoncom
    bridge=PowerPointComBridge()
    office,initialized=bridge._new_application()
    presentation=None
    try:
        presentation=office.Presentations.Open(str(output),ReadOnly=True,Untitled=False,WithWindow=False)
        record['reopened_slide_count']=presentation.Slides.Count
        for index in range(1,presentation.Slides.Count+1):
            presentation.Slides.Item(index).Export(str(p.AUDIT/'screens'/f'powerpoint-export-{index}.png'),'PNG',1600,900)
    finally:
        if presentation is not None: presentation.Close()
        presentation=None
        office=None
        if initialized: pythoncom.CoUninitialize()
    record['status']='PASS'
except Exception:
    record['status']='FAIL'
    record['error']=traceback.format_exc()
finally:
    (p.AUDIT/'powerpoint-results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2),flush=True)
    p.window.hide()
p.sys.exit(0 if record['status'] == 'PASS' else 1)
