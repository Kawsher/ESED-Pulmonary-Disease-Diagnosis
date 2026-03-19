# =============================================================
# base_learner_training.py — Two-Phase Transfer Learning
# =============================================================
# Trains 5 CNN base learners using two-phase transfer learning:
#   Phase 1: Frozen base, train head only
#   Phase 2: Unfreeze last 30 layers, fine-tune
#
# Models: DenseNet201, EfficientNetB4, ResNet50V2,
#         InceptionV3, ConvNeXtTiny
# =============================================================

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import (
    DenseNet201, EfficientNetB4, ResNet50V2,
    InceptionV3, ConvNeXtTiny
)
from tensorflow.keras.applications.densenet     import preprocess_input as densenet_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_pre
from tensorflow.keras.applications.resnet_v2    import preprocess_input as resnet_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_pre
from tensorflow.keras.applications.convnext     import preprocess_input as convnext_pre
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)

from config import (
    CLASSES, NUM_CLASSES, SEED,
    IMG_SIZE_DEFAULT, IMG_SIZE_INCEPTION,
    BATCH_SIZE, PHASE1_EPOCHS, PHASE2_EPOCHS,
    PHASE1_LR, PHASE2_LR, UNFREEZE_LAYERS,
    DROPOUT_RATE_1, DROPOUT_RATE_2,
    DENSE_UNITS_1, DENSE_UNITS_2,
    EARLY_STOP_PATIENCE, LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE, MODELS_DIR, LOGS_DIR,
    GRADCAM_LAYERS
)

# Set seeds
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── Preprocessing functions lookup ────────────────────────────
PREPROCESS_FNS = {
    'DenseNet201'   : densenet_pre,
    'EfficientNetB4': efficientnet_pre,
    'ResNet50V2'    : resnet_pre,
    'InceptionV3'   : inception_pre,
    'ConvNeXtTiny'  : convnext_pre,
}

IMG_SIZES = {
    'DenseNet201'   : IMG_SIZE_DEFAULT,
    'EfficientNetB4': IMG_SIZE_DEFAULT,
    'ResNet50V2'    : IMG_SIZE_DEFAULT,
    'InceptionV3'   : IMG_SIZE_INCEPTION,
    'ConvNeXtTiny'  : IMG_SIZE_DEFAULT,
}


# ── Model Architecture ────────────────────────────────────────

def build_base_model(model_name: str,
                      img_size  : int
                      ) -> keras.Model:
    """
    Load pretrained base with ImageNet weights.

    Args:
        model_name: One of the 5 supported architectures
        img_size  : Input image size

    Returns:
        Keras Model (frozen base)
    """
    input_shape = (img_size, img_size, 3)
    kwargs      = dict(
        weights    = 'imagenet',
        include_top= False,
        input_shape= input_shape)

    base_map = {
        'DenseNet201'   : DenseNet201,
        'EfficientNetB4': EfficientNetB4,
        'ResNet50V2'    : ResNet50V2,
        'InceptionV3'   : InceptionV3,
        'ConvNeXtTiny'  : ConvNeXtTiny,
    }
    base = base_map[model_name](**kwargs)
    base.trainable = False
    return base


def build_classification_head(
        base       : keras.Model,
        model_name : str
) -> keras.Model:
    """
    Add custom classification head to pretrained base.
    ConvNeXtTiny uses LayerNorm + GELU + AdamW.
    Others use BatchNorm + ReLU + Adam.

    Returns:
        Full compiled Keras Model
    """
    is_convnext = model_name == 'ConvNeXtTiny'

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)

    if is_convnext:
        x = layers.LayerNormalization()(x)
    else:
        x = layers.BatchNormalization()(x)

    x = layers.Dense(
        DENSE_UNITS_1,
        activation='gelu' if is_convnext else 'relu',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(DROPOUT_RATE_1)(x)

    if is_convnext:
        x = layers.LayerNormalization()(x)
    else:
        x = layers.BatchNormalization()(x)

    x = layers.Dense(
        DENSE_UNITS_2,
        activation='gelu' if is_convnext else 'relu',
        kernel_regularizer=keras.regularizers.l2(1e-4)
    )(x)
    x = layers.Dropout(DROPOUT_RATE_2)(x)

    output = layers.Dense(
        NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(
        inputs=base.input, outputs=output)
    return model


def compile_model(model     : keras.Model,
                   model_name: str,
                   lr        : float) -> keras.Model:
    """Compile model with appropriate optimiser."""
    if model_name == 'ConvNeXtTiny':
        optimiser = keras.optimizers.AdamW(
            learning_rate=lr, weight_decay=1e-4)
    else:
        optimiser = keras.optimizers.Adam(
            learning_rate=lr)

    model.compile(
        optimizer = optimiser,
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'])
    return model


# ── Callbacks ─────────────────────────────────────────────────

def get_callbacks(model_name  : str,
                   phase       : int,
                   monitor     : str = 'val_loss'
                   ) -> list:
    """
    Get training callbacks for a given phase.

    Args:
        model_name: Model name for file naming
        phase     : 1 or 2
        monitor   : Metric to monitor

    Returns:
        List of Keras callbacks
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,   exist_ok=True)

    ckpt_path = (MODELS_DIR +
                 f'{model_name}_phase{phase}_best.keras')

    return [
        EarlyStopping(
            monitor             = monitor,
            patience            = EARLY_STOP_PATIENCE,
            restore_best_weights= True,
            verbose             = 1),
        ModelCheckpoint(
            filepath        = ckpt_path,
            monitor         = monitor,
            save_best_only  = True,
            verbose         = 1),
        ReduceLROnPlateau(
            monitor  = monitor,
            factor   = LR_REDUCE_FACTOR,
            patience = LR_REDUCE_PATIENCE,
            min_lr   = 1e-7,
            verbose  = 1),
        CSVLogger(
            LOGS_DIR +
            f'{model_name}_phase{phase}_history.csv'),
    ]


# ── Two-Phase Training ────────────────────────────────────────

def train_phase1(model      : keras.Model,
                  model_name : str,
                  train_gen  ,
                  val_gen    ,
                  class_weights: dict
                  ) -> keras.callbacks.History:
    """
    Phase 1: Train classification head with frozen base.

    Returns:
        Training history
    """
    print(f"\n{'='*55}")
    print(f"[{model_name}] Phase 1 — Head training")
    print(f"  LR={PHASE1_LR} | Epochs={PHASE1_EPOCHS}")
    print(f"{'='*55}")

    model    = compile_model(model, model_name, PHASE1_LR)
    callbacks= get_callbacks(model_name, phase=1)

    history  = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = PHASE1_EPOCHS,
        class_weight    = class_weights,
        callbacks       = callbacks,
        verbose         = 1,
    )
    return history


def train_phase2(model      : keras.Model,
                  model_name : str,
                  train_gen  ,
                  val_gen    ,
                  class_weights: dict
                  ) -> keras.callbacks.History:
    """
    Phase 2: Unfreeze last 30 layers and fine-tune.

    Returns:
        Training history
    """
    print(f"\n{'='*55}")
    print(f"[{model_name}] Phase 2 — Fine-tuning")
    print(f"  LR={PHASE2_LR} | Epochs={PHASE2_EPOCHS}")
    print(f"  Unfreezing last {UNFREEZE_LAYERS} layers")
    print(f"{'='*55}")

    # Unfreeze last N layers
    for layer in model.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    n_trainable = sum(
        1 for l in model.layers if l.trainable)
    print(f"  Trainable layers: {n_trainable}")

    model     = compile_model(model, model_name, PHASE2_LR)
    callbacks = get_callbacks(model_name, phase=2)

    history   = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = PHASE2_EPOCHS,
        class_weight    = class_weights,
        callbacks       = callbacks,
        verbose         = 1,
    )
    return history


def train_model(model_name   : str,
                 train_df     ,
                 val_df       ,
                 class_weights: dict
                 ) -> tuple[keras.Model, dict]:
    """
    Full two-phase training pipeline for one model.

    Args:
        model_name   : Architecture name
        train_df     : Training DataFrame
        val_df       : Validation DataFrame
        class_weights: Dict of class weights

    Returns:
        (trained_model, combined_history_dict)
    """
    from dataset_preparation import (
        get_train_generator, get_val_generator)

    img_size = IMG_SIZES[model_name]
    pre_fn   = PREPROCESS_FNS[model_name]

    # Data generators
    train_gen = get_train_generator(
        train_df, pre_fn, img_size, augment=True)
    val_gen   = get_val_generator(
        val_df, pre_fn, img_size)

    # Build model
    base  = build_base_model(model_name, img_size)
    model = build_classification_head(base, model_name)

    print(f"\n[{model_name}] Total params: "
          f"{model.count_params():,}")

    # Phase 1
    hist1 = train_phase1(
        model, model_name, train_gen,
        val_gen, class_weights)

    # Phase 2
    hist2 = train_phase2(
        model, model_name, train_gen,
        val_gen, class_weights)

    # Save final model
    save_path = MODELS_DIR + f'{model_name}_final.keras'
    model.save(save_path)
    print(f"\n✅ Saved: {save_path}")

    # Combine histories
    combined = {
        'phase1': hist1.history,
        'phase2': hist2.history,
    }

    return model, combined


def train_all_models(train_df     ,
                      val_df       ,
                      class_weights: dict
                      ) -> dict:
    """
    Train all 5 base learners sequentially.

    Returns:
        Dict of training histories per model
    """
    model_names = [
        'DenseNet201', 'EfficientNetB4',
        'ResNet50V2', 'InceptionV3', 'ConvNeXtTiny']

    all_histories = {}

    for model_name in model_names:
        print(f"\n{'#'*60}")
        print(f"Training: {model_name}")
        print(f"{'#'*60}")

        _, history = train_model(
            model_name, train_df, val_df, class_weights)
        all_histories[model_name] = history

        # Free GPU memory
        import gc
        tf.keras.backend.clear_session()
        gc.collect()

    print("\n✅ All 5 models trained successfully")
    return all_histories


if __name__ == '__main__':
    print("Base learner training module loaded.")
    print(f"Models: DenseNet201, EfficientNetB4, "
          f"ResNet50V2, InceptionV3, ConvNeXtTiny")
    print(f"Phase 1: {PHASE1_EPOCHS} epochs @ LR={PHASE1_LR}")
    print(f"Phase 2: {PHASE2_EPOCHS} epochs @ LR={PHASE2_LR}")
