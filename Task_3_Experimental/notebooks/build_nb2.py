import json

def code_cell(source):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [source]
    }

cells = []

# Cell 1: Imports + GPU
cells.append(code_cell(
"""import tensorflow as tf
import numpy as np, pandas as pd, cv2, time
from pathlib import Path
import os
import matplotlib.pyplot as plt

os.environ['TF_CUDNN_USE_AUTOTUNE']       = '0'
os.environ['TF_XLA_FLAGS']                = '--tf_xla_auto_jit=0'
os.environ['TF_DISABLE_MKL']              = '1'
os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
tf.config.optimizer.set_jit(False)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f'GPU: {gpus[0].name}')
else:
    print('WARNING: No GPU found — running on CPU')
"""
))

# Cell 2: Config
cells.append(code_cell(
"""DATA_ROOT = Path('/home/krish/vein_detection_task_3_training')
VEIN_DIR  = DATA_ROOT / 'output/vein'
CKPT_DIR  = VEIN_DIR / 'checkpoints'
PLOT_DIR  = VEIN_DIR / 'plots'
CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 60
LR         = 1e-4
N_CLASSES  = 2
VAL_FRAC   = 0.15
SEED       = 42

print('Config OK')
print(f'  IMG_SIZE={IMG_SIZE}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}')
"""
))

# Cell 3: Load + validate metadata
cells.append(code_cell(
"""df_all  = pd.read_csv(VEIN_DIR / 'metadata.csv')
df_vein = df_all[df_all['has_vein']].reset_index(drop=True)
print(f'Total frames : {len(df_all)}')
print(f'With veins   : {len(df_vein)}')

# Fix relative paths to absolute
df_vein['frame_path'] = df_vein['frame_path'].apply(
    lambda p: str(DATA_ROOT / p) if not p.startswith('/') else p)
df_vein['mask_path'] = df_vein['mask_path'].apply(
    lambda p: str(DATA_ROOT / p) if not p.startswith('/') else p)

print('Validating files...')
valid = []
for _, row in df_vein.iterrows():
    f = cv2.imread(row['frame_path'])
    m = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
    if f is not None and m is not None and f.shape[:2] == m.shape[:2]:
        valid.append(row)
df_clean = pd.DataFrame(valid).reset_index(drop=True)
print(f'Valid pairs  : {len(df_clean)} / {len(df_vein)}')
"""
))

# Cell 4: Train/val split
cells.append(code_cell(
"""from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    df_clean, test_size=VAL_FRAC, random_state=SEED, shuffle=True)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

print(f'Train: {len(train_df)}  |  Val: {len(val_df)}')
"""
))

# Cell 5: Dataset pipeline
cells.append(code_cell(
"""MEAN_NP = np.array([0.485, 0.456, 0.406], np.float32)
STD_NP  = np.array([0.229, 0.224, 0.225], np.float32)

def _load_pair(fp, mp):
    img  = cv2.cvtColor(cv2.imread(fp.decode()), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    mask = cv2.imread(mp.decode(), cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.int32)
    return img, mask

def make_dataset(dataframe, batch_size, shuffle=False):
    fps = dataframe['frame_path'].values
    mps = dataframe['mask_path'].values
    def load(fp, mp):
        img, mask = tf.numpy_function(_load_pair, [fp, mp], [tf.float32, tf.int32])
        img  = (img - MEAN_NP) / STD_NP
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        mask.set_shape([IMG_SIZE, IMG_SIZE])
        return img, mask
    ds = tf.data.Dataset.from_tensor_slices((fps, mps))
    if shuffle:
        ds = ds.shuffle(len(dataframe), seed=SEED)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_df, BATCH_SIZE, shuffle=True)
val_ds   = make_dataset(val_df,   BATCH_SIZE, shuffle=False)

for xb, yb in train_ds.take(1):
    print(f'Image batch : {xb.shape} {xb.dtype}')
    print(f'Mask  batch : {yb.shape} {yb.dtype}  unique={np.unique(yb.numpy())}')
    vein_pct = float(tf.reduce_mean(tf.cast(yb == 1, tf.float32))) * 100
    print(f'Vein pixel %: {vein_pct:.2f}%')
"""
))

# Cell 6: Model definition
cells.append(code_cell(
"""CW_VEIN = tf.constant([0.5, 15.0], dtype=tf.float32)

def vein_dice_ce_loss(y_true, y_pred):
    smooth   = 1e-6
    y_true_i = tf.cast(y_true, tf.int32)
    y_oh     = tf.one_hot(y_true_i, N_CLASSES)
    inter    = tf.reduce_sum(y_pred * y_oh, axis=[0, 1, 2])
    union    = tf.reduce_sum(y_pred + y_oh, axis=[0, 1, 2])
    dice     = 1. - tf.reduce_mean((2. * inter + smooth) / (union + smooth))
    ce       = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce       = tf.reduce_mean(ce * tf.gather(CW_VEIN, y_true_i))
    return 0.5 * dice + 0.5 * ce

def vein_iou(y_true, y_pred):
    pc = tf.cast(tf.argmax(y_pred, -1), tf.int32)
    tc = tf.cast(y_true, tf.int32)
    tp = tf.reduce_sum(tf.cast(tf.equal(pc, 1) & tf.equal(tc, 1),     tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.equal(pc, 1) & tf.not_equal(tc, 1), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.not_equal(pc, 1) & tf.equal(tc, 1), tf.float32))
    return tp / (tp + fp + fn + 1e-6)

def vein_dice(y_true, y_pred):
    pc = tf.cast(tf.argmax(y_pred, -1) == 1, tf.float32)
    tc = tf.cast(tf.cast(y_true, tf.int32) == 1, tf.float32)
    return 2. * tf.reduce_sum(pc * tc) / (tf.reduce_sum(pc) + tf.reduce_sum(tc) + 1e-6)

def conv_block(x, f):
    for _ in range(2):
        x = tf.keras.layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x

def build_unet(n_classes=N_CLASSES):
    inp  = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3))
    base = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_tensor=inp)
    s1 = base.get_layer('conv1_relu').output
    s2 = base.get_layer('conv2_block3_out').output
    s3 = base.get_layer('conv3_block4_out').output
    s4 = base.get_layer('conv4_block6_out').output
    x  = base.get_layer('conv5_block3_out').output
    for skip, f in [(s4, 256), (s3, 128), (s2, 64), (s1, 32)]:
        x = tf.keras.layers.UpSampling2D(2)(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        x = conv_block(x, f)
    x   = tf.keras.layers.UpSampling2D(2)(x)
    x   = conv_block(x, 16)
    out = tf.keras.layers.Conv2D(n_classes, 1, activation='softmax')(x)
    return tf.keras.Model(inp, out)

model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss=vein_dice_ce_loss,
    metrics=[vein_iou, vein_dice]
)
model.summary(line_length=80)
"""
))

# Cell 7: Training
cells.append(code_cell(
"""CKPT_PATH = str(CKPT_DIR / 'unet_resnet50_vein_best.keras')

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH,
        monitor='val_vein_iou', mode='max',
        save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_vein_iou', mode='max',
        factor=0.5, patience=8, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_vein_iou', mode='max',
        patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.CSVLogger(
        str(VEIN_DIR / 'training_log.csv'), append=True),
]

t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)
print(f'Training done in {(time.time()-t0)/60:.1f} min')
"""
))

# Cell 8: Training curves
cells.append(code_cell(
"""log_df = pd.read_csv(VEIN_DIR / 'training_log.csv')

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
pairs = [
    ('loss',      'val_loss',      'Loss'),
    ('vein_iou',  'val_vein_iou',  'Vein IoU'),
    ('vein_dice', 'val_vein_dice', 'Vein Dice'),
]
for ax, (tr, vl, title) in zip(axes, pairs):
    ax.plot(log_df['epoch'], log_df[tr], label='Train')
    ax.plot(log_df['epoch'], log_df[vl], label='Val')
    ax.set_title(title); ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(str(PLOT_DIR / 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()
best_ep  = int(log_df['val_vein_iou'].idxmax())
best_iou = float(log_df['val_vein_iou'].max())
print(f'Best val_vein_iou: {best_iou:.4f}  at epoch {best_ep}')
"""
))

# Cell 9: Visual predictions
cells.append(code_cell(
"""best_model = tf.keras.models.load_model(
    CKPT_PATH,
    custom_objects={
        'vein_dice_ce_loss': vein_dice_ce_loss,
        'vein_iou': vein_iou,
        'vein_dice': vein_dice
    })

print('Evaluating on validation set...')
results = best_model.evaluate(val_ds, verbose=1)
print(dict(zip(best_model.metrics_names, results)))

n_show = 6
sample = val_df.sample(n_show, random_state=SEED).reset_index(drop=True)

fig, axes = plt.subplots(n_show, 3, figsize=(10, n_show * 3))
for i, row in sample.iterrows():
    raw  = cv2.cvtColor(cv2.imread(row['frame_path']), cv2.COLOR_BGR2RGB)
    raw  = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
    gt   = (cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
    inp  = ((raw.astype(np.float32) / 255. - MEAN_NP) / STD_NP)[None]
    pred = best_model.predict(inp, verbose=0)[0]
    pred_mask = np.argmax(pred, -1).astype(np.uint8)

    axes[i, 0].imshow(raw);               axes[i, 0].set_title('Frame')
    axes[i, 1].imshow(gt,        cmap='gray'); axes[i, 1].set_title('GT Mask')
    axes[i, 2].imshow(pred_mask, cmap='gray'); axes[i, 2].set_title('Predicted')
    for ax in axes[i]: ax.axis('off')

plt.tight_layout()
plt.savefig(str(PLOT_DIR / 'sample_predictions.png'), dpi=120, bbox_inches='tight')
plt.show()
"""
))

# Cell 10: Per-frame metrics + final summary
cells.append(code_cell(
"""ious, dices = [], []
for _, row in val_df.iterrows():
    raw  = cv2.cvtColor(cv2.imread(row['frame_path']), cv2.COLOR_BGR2RGB)
    raw  = cv2.resize(raw, (IMG_SIZE, IMG_SIZE))
    gt   = (cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE) > 0).astype(np.int32).flatten()
    inp  = ((raw.astype(np.float32) / 255. - MEAN_NP) / STD_NP)[None]
    pred = np.argmax(best_model.predict(inp, verbose=0)[0], -1).flatten()
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    ious.append(tp / (tp + fp + fn + 1e-6))
    dices.append(2*tp / (2*tp + fp + fn + 1e-6))

print('=' * 47)
print('  VEIN DETECTION — VALIDATION SUMMARY')
print('=' * 47)
print(f'  Mean IoU  : {np.mean(ious):.4f}  +/- {np.std(ious):.4f}')
print(f'  Mean Dice : {np.mean(dices):.4f}  +/- {np.std(dices):.4f}')
print(f'  Median IoU: {np.median(ious):.4f}')
print(f'  IoU > 0.5 : {np.mean(np.array(ious) > 0.5)*100:.1f}% of frames')
print(f'  Checkpoint: {CKPT_PATH}')
print('=' * 47)
"""
))

nb = {
    'nbformat': 4,
    'nbformat_minor': 5,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {'name': 'python', 'version': '3.13.0'}
    },
    'cells': cells
}

with open('/home/krish/vein_detection_task_3_training/02_vein_detection.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print(f'Written 02_vein_detection.ipynb — {len(cells)} cells')
