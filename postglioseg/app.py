"""PostGlioSeg, автор - Никишев Иван Олегович"""
from typing import Any
import os, shutil
from functools import partial
import tempfile

from tkinter import Tk, filedialog
import gradio as gr

import numpy as np
import torch

import SimpleITK as sitk
from pydicom import dcmread
import highdicom as hd

from .python_tools import find_file_containing
from .mri.pipeline import Pipeline
from .mri.registration import resample_to
from .inference import tta, save_slices
from .model import get_pretrained_model
from .dicom_segment_descriptions import SEGMENTS

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL = get_pretrained_model().to(DEVICE)

def _get_path(path, mod) -> str:
    """File either `{mod}.ext` or a {mod} folder."""
    p = find_file_containing(path, f'{mod}.', lower=True, error=False)
    if p is None: p = os.path.join(path, mod)
    if os.path.exists(p): return p
    return None # type:ignore

def dcmreadfolder(path):
    dicom_files = []
    for i in os.listdir(path):
        try:
            dicom_files.append(dcmread(os.path.join(path, i)))
        except Exception as e: pass
    return sorted(dicom_files, key=lambda x: x.InstanceNumber)


def inference(
    folder: str,
    register: bool,
    skullstrip: bool,
    save_sri: bool,
    #save_nifti: bool,
    save_jpeg: bool,
    save_segmentation_on: list[str],
    workdir,
    _progress=gr.Progress(True),
):
    try:
        #print(folder)
        if not folder: raise gr.Error('Не указана директория.')
        #folder = load_folder()
        #print(gr_folder)
        #print(register, skullstrip, save_jpeg, save_segmentation_on)
        #folder = gr_folder
        _progress(0, "Загрузка файлов...")
        tempdir = os.path.join(workdir, 'temp')
        if os.path.exists(tempdir): shutil.rmtree(tempdir)
        os.mkdir(tempdir)

        # get paths
        t1 = _get_path(folder, 't1')
        t1ce = _get_path(folder, 't1ce')
        flair = _get_path(folder, 'flair')
        t2 = _get_path(folder, 't2')
        seg = _get_path(folder, 'seg')
        
        if t1 is None: raise gr.Error('Не найдена модальность t1. В директории дожна быть поддиректория "t1" или файл "t1.nii.gz"')
        if t1ce is None: raise gr.Error('Не найдена модальность t1ce. В директории дожна быть поддиректория "t1ce" или файл "t1ce.nii.gz"')
        if flair is None: raise gr.Error('Не найдена модальность flair. В директории дожна быть поддиректория "flair" или файл "flair.nii.gz"')
        if t2 is None: raise gr.Error('Не найдена модальность t2. В директории дожна быть поддиректория "t2" или файл "t2.nii.gz"')

        mod_dict = dict(T1=t1, T1CE=t1ce, FLAIR=flair, T2=t2)

        #print(t1, t1ce, flair, t2, seg)

        # preprocess files
        _progress(0.05, "Загрузка файлов...")
        pipeline = Pipeline(t1=t1, t1ce=t1ce, flair=flair, t2w=t2)
        _progress(0.05, "Предобработка файлов")
        inputs = pipeline.preprocess(register=register, skullstrip=skullstrip, _progress=_progress)

        # run prediction
        with torch.no_grad():
            MODEL.eval()
            preds = tta(inputs = inputs.to(DEVICE), inferer = MODEL,  size = (96,96), overlap=0.5, _progress=_progress).argmax(0)

        # convert preds to sitk and add the deleted background to match SRI24
        preds_sitk = sitk.GetImageFromArray(preds.cpu().numpy().astype(np.float32))
        preds_sitk.CopyInformation(pipeline.t1ce_final)
        preds_sitk = resample_to(preds_sitk, pipeline.t1ce_sri)

        # write preds in sri
        if save_sri:
            _progress(0.75, "Сохранение сегментации в NIfTI...")
            sitk.WriteImage(preds_sitk, os.path.join(tempdir, 'SRI24_segmentations.nii.gz'))
            sitk.WriteImage(resample_to(pipeline.t1_final, pipeline.t1_sri), os.path.join(tempdir, 'SRI24_t1.nii.gz'))
            sitk.WriteImage(resample_to(pipeline.t1ce_final, pipeline.t1ce_sri), os.path.join(tempdir, 'SRI24_t1ce.nii.gz'))
            sitk.WriteImage(resample_to(pipeline.flair_final, pipeline.flair_sri), os.path.join(tempdir, 'SRI24_flair.nii.gz'))
            sitk.WriteImage(resample_to(pipeline.t2w_final, pipeline.t2w_sri), os.path.join(tempdir, 'SRI24_t2.nii.gz'))

        # register preds to native modalities and write them
        if register and len(save_segmentation_on) > 0:
            for i,mod in enumerate(save_segmentation_on):
                _progress(0.85+i*0.02, f"Совмещение сегментации к {mod}...")
                preds_native = pipeline.postprocess(preds, to=mod)
                _progress(0.85+i*0.02+0.01, f"Сохранение сегментации на {mod}...")
                sitk.WriteImage(preds_native, os.path.join(tempdir, f'{mod}_seg_native.nii.gz'))

                # save DICOM-seg if original image is DICOM
                if os.path.isdir(mod_dict[mod]):
                    source_images = dcmreadfolder(mod_dict[mod])
                    #print(source_images)
                    #print([i.InstanceNumber for i in source_images])

                    # All of those are necessary for DICOM to work
                    seg_dicom = hd.seg.Segmentation(
                        source_images=source_images,

                        pixel_array=np.flip(sitk.GetArrayFromImage(preds_native).astype(np.uint16), (1,)) if "T1" not in mod
                        else np.flip(sitk.GetArrayFromImage(preds_native).astype(np.uint16), (1,2)),  # sitk flips arrays

                        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
                        segment_descriptions=SEGMENTS,
                        series_instance_uid=source_images[0].SeriesInstanceUID,
                        series_number=source_images[0].SeriesNumber,
                        sop_instance_uid=source_images[0].SOPInstanceUID,
                        instance_number=source_images[0].InstanceNumber,
                        manufacturer=source_images[0].Manufacturer,
                        manufacturer_model_name=source_images[0].ManufacturerModelName,
                        software_versions=source_images[0].SoftwareVersions,
                        device_serial_number=source_images[0].DeviceSerialNumber,
                    )

                    seg_dicom.save_as(os.path.join(tempdir, f'{mod}_seg_native.dcm'))

        # save jpeg slices
        _progress(0.9, "Сохранение срезов сегментации в JPEG..")
        if save_jpeg:
            os.mkdir(os.path.join(tempdir, 'срезы'))
            save_slices(
                (pipeline.t1_sri, pipeline.t1ce_sri, pipeline.flair_sri, pipeline.t2w_sri),
                preds_sitk,
                os.path.join(tempdir, "срезы"),
                real_seg=seg,
            )

        _progress(0.95, "Архивирование файлов..")
        if os.path.exists(os.path.join(workdir, f'{os.path.basename(folder)}.zip')): os.remove(os.path.join(workdir, f'{os.path.basename(folder)}.zip'))
        shutil.make_archive(os.path.join(workdir,  f'{os.path.basename(folder)}'), 'zip', tempdir)

        # return zip
        return os.path.join(workdir, f'{os.path.basename(folder)}.zip')

    except Exception as e:
        raise gr.Error(str(e))

def select_folder():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filename = filedialog.askdirectory()
    root.destroy()
    if filename: return filename
    else: raise gr.Error("Директория не выбрана.")


def interface(tempdir):
    with gr.Blocks() as ui:
        with gr.Row():
            with gr.Column():
                image_browse_btn = gr.Button("Выберите директорию", min_width=1)
                input_path = gr.Textbox(label="Выбранная директория", scale=5, interactive=False)
                register = gr.Checkbox(label="Произвести регистрацию к атласу SRI24", value=True)
                skullstrip = gr.Checkbox(label="Удалить череп", value=True)
                save_sri = gr.Checkbox(label="Сохранить обработанное обследование", value=True)
                #save_nifti = gr.Checkbox(label="Также сохранить сегментацию в NIfTI", value=True)
                save_jpeg = gr.Checkbox(label="Сохранить срезы в JPEG", value=True)
                save_segmentation_on = gr.CheckboxGroup(label="Сохранить сегментацию на", choices=["T1", "T1CE", "FLAIR", "T2"], value=[])
                inference_btn = gr.Button("Сегментация", variant="primary")
            with gr.Column():
                file = gr.File(label="Сегментация", type='filepath', interactive=False)
        image_browse_btn.click(select_folder, outputs=input_path, show_progress="hidden") # type:ignore #pylint:disable=E1101
        inference_btn.click(partial(inference, workdir = tempdir), inputs=[input_path, register, skullstrip, save_sri,  # type:ignore #pylint:disable=E1101
                                                                           #save_nifti,
                                                                           save_jpeg, save_segmentation_on], outputs=file) # type:ignore #pylint:disable=E1101
    return ui

def run():
    with tempfile.TemporaryDirectory() as tempdir:
        ui = interface(tempdir)
        ui.queue().launch(inbrowser=True)
