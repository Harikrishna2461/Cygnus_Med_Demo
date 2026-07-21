"""
Standalone training launcher — run from anywhere:
    python run_training.py
Output streams directly to terminal (tqdm bars live inside the subprocess).
"""
import subprocess, sys, os
from pathlib import Path

BASE_DIR    = Path(__file__).parent
BIOMEDPARSE = BASE_DIR / 'BiomedParse'

env = os.environ.copy()
env['PYTHONPATH']       = str(BASE_DIR / 'stubs') + os.pathsep + str(BIOMEDPARSE) + os.pathsep + env.get('PYTHONPATH', '')
env['DETECTRON2_DATASETS'] = str(BIOMEDPARSE / 'biomedparse_datasets') + os.sep
env['DATASET']          = str(BIOMEDPARSE / 'biomedparse_datasets') + os.sep
env['DATASET2']         = str(BIOMEDPARSE / 'biomedparse_datasets') + os.sep
env['VLDATASET']        = str(BIOMEDPARSE / 'biomedparse_datasets') + os.sep
env['OMPI_ALLOW_RUN_AS_ROOT']         = '1'
env['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
env['PYTHONWARNINGS']   = 'ignore'
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['PYTHONUNBUFFERED'] = '1'

cmd = [
    sys.executable, '-u', 'entry.py', 'train',
    '--conf_files', 'configs/biomed_fascia_finetuning.yaml',
    '--overrides',
    'FP16', 'True',
    'RANDOM_SEED', '2024',
    'BioMed.INPUT.IMAGE_SIZE', '512',
    'MODEL.DECODER.HIDDEN_DIM', '512',
    'MODEL.ENCODER.CONVS_DIM', '512',
    'MODEL.ENCODER.MASK_DIM', '512',
    'TEST.BATCH_SIZE_TOTAL', '2',
    'TRAIN.BATCH_SIZE_TOTAL', '2',
    'TRAIN.BATCH_SIZE_PER_GPU', '2',
    'SOLVER.MAX_NUM_EPOCHS', '2',
    'SOLVER.BASE_LR', '0.00001',
    'SOLVER.FIX_PARAM.backbone', 'True',
    'SOLVER.FIX_PARAM.lang_encoder', 'True',
    'SOLVER.FIX_PARAM.pixel_decoder', 'False',
    'MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT', '1.0',
    'MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT',  '1.0',
    'MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT',  '1.0',
    'MODEL.DECODER.TOP_SPATIAL_LAYERS', '10',
    'MODEL.DECODER.SPATIAL.ENABLED', 'True',
    'MODEL.DECODER.GROUNDING.ENABLED', 'True',
    'MODEL.DECODER.SPATIAL.MAX_ITER', '0',
    'LOADER.SAMPLE_PROB', 'prop',
    'BioMed.INPUT.RANDOM_ROTATE', 'True',
    'FIND_UNUSED_PARAMETERS', 'True',
    'ATTENTION_ARCH.SPATIAL_MEMORIES', '32',
    'ATTENTION_ARCH.QUERY_NUMBER', '3',
    'STROKE_SAMPLER.MAX_CANDIDATE', '10',
    'WEIGHT', 'True',
    'LOG_EVERY', '1',
    'RESUME_FROM', str(BASE_DIR / 'pretrained' / 'biomedparse_v1.pt'),
    'SAVE_DIR', 'output/fascia_finetuning_v2',
]

print("=" * 70)
print("BiomedParse fascia fine-tuning")
print("  512px | batch=2 | 5 epochs | pixel_decoder TRAINABLE")
print("  Checkpoints → BiomedParse/output/fascia_finetuning_v2/")
print("  First step takes 1-3 min (CUDA kernel compilation)")
print("=" * 70)
print(flush=True)

proc = subprocess.Popen(
    cmd,
    cwd=str(BIOMEDPARSE),
    env=env,
    stdout=None,   # inherit terminal stdout → tqdm bars render live
    stderr=None,   # inherit terminal stderr
)

proc.wait()
print(f"\nDone — exit code {proc.returncode}")