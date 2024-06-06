"""PostGlioSeg, автор - Никишев Иван Олегович"""
from typing import Any
import os
import tempfile, zipfile
from tkinter import Tk, filedialog
import io
import torch
import SimpleITK as sitk
import gradio as gr
from .python_tools import find_file_containing
from .mri.pipeline import Pipeline
from .inference import tta, save_slices
from .model import get_pretrained_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL = get_pretrained_model().to(DEVICE)

def _get_path(path, mod) -> str:
    """File either `{mod}.ext` or a {mod} folder."""
    p = find_file_containing(path, f'{mod}.', lower=True, error=False)
    if p is None: p = os.path.join(path, mod)
    if os.path.isdir(p): return p
    return None # type:ignore


def load_folder():
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    filename = filedialog.askdirectory()
    if filename:
        root.destroy()
        return filename
    else:
        root.destroy()
        raise gr.Error("Не указана директория")


def inference(folder, register:bool, skullstrip:bool, save_jpeg:bool, save_segmentation_on:list[str], _progress = gr.Progress(True)):
    #folder = load_folder()
    print(folder)
    with tempfile.TemporaryDirectory() as tempdir:
        _progress(0, "Поиск директорий...")
        # get paths
        t1 = _get_path(folder, 't1')
        t1ce = _get_path(folder, 't1ce')
        flair = _get_path(folder, 'flair')
        t2 = _get_path(folder, 't2')
        seg = _get_path(folder, 'seg')
        
        print(t1, t1ce, flair, t2, seg)

        # preprocess files
        pipeline = Pipeline(t1=t1, t1ce=t1ce, flair=flair, t2w=t2)
        _progress(5, "Предобработка файлов")
        inputs = pipeline.preprocess(register=register, skullstrip=skullstrip, _progress=_progress)

        # run prediction
        preds = tta(inputs = inputs.to(DEVICE), inferer = MODEL,  size = (96,96), overlap=0.75, _progress=_progress).argmax(0)

        # convert preds to sitk
        preds_sitk = sitk.GetImageFromArray(preds.cpu().numpy())
        preds_sitk.CopyInformation(pipeline.t1ce_final)
        # write preds
        _progress(85, "Сохранение сегментации в NIfTI...")
        sitk.WriteImage(preds_sitk, os.path.join(tempdir, 'сегментация в SRI24.nii.gz'))

        # register preds to native modalities and write them
        if register and len(save_segmentation_on) > 0:
            for i,mod in enumerate(save_segmentation_on):
                _progress(85+i*2, f"Совмещение регистрации к {mod}...")
                preds_native = pipeline.postprocess(preds, to=mod)
                _progress(85+i*2+1, f"Сохранение сегментации на {mod}...")
                sitk.WriteImage(preds_native, os.path.join(tempdir, f'сегментация {mod}.nii.gz'))

        # save jpeg slices
        _progress(90, "Сохранение срезов сегментации в JPEG..")
        if save_jpeg:
            os.mkdir(os.path.join(tempdir, 'срезы'))
            save_slices(
                (pipeline.t1_sri, pipeline.t1ce_sri, pipeline.flair_sri, pipeline.t2w_sri),
                preds_sitk,
                os.path.join(tempdir, "срезы"),
                real_seg=seg,
            )

        # zip the folder
        _progress(95, "Сжатие файлов...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip:
            for root, dirs, files in os.walk(tempdir):
                for file in files:
                    zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(tempdir, '..')))

        # return zip bytes
        return zip_buffer.getvalue()




def run():
    interface = gr.Interface(
        inference,
        [
            gr.FileExplorer(root_dir= os.path.abspath('.').split(os.path.sep)[0]+os.path.sep, label="Выберите директорию", file_count='single'),
            gr.Checkbox(label = 'Произвести совмещение к атласу SRI24'),
            gr.Checkbox(label = 'Произвести удаление черепа'),
            gr.Checkbox(label = 'Сохранить срезы'),
            gr.CheckboxGroup(choices = ['t1', 't1ce', 'flair', 't2'], label = 'Сохранить сегментацию на'),
        ],
        [
            gr.File(type="binary")
        ],
        title = 'Автор - Никишев И.О.',
        description = '''''',
        article='''''',
        allow_flagging = 'never',
        theme = gr.themes.Soft(),
    )

    interface.queue().launch()