# =============================================================
# xai_analysis.py — Explainability Analysis
# =============================================================
# Implements three complementary XAI methods:
#   1. Grad-CAM  — gradient-based, all 5 models
#   2. SHAP      — game-theoretic, DenseNet201
#   3. LIME      — perturbation-based, DenseNet201
#
# Also computes:
#   - XAI agreement (Grad-CAM vs SHAP spatial correlation)
#   - Region importance analysis
#   - LIME quantitative superpixel analysis
# =============================================================

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import (
    load_img, img_to_array)
from scipy.stats import spearmanr, pearsonr, entropy

from config import (
    CLASSES, NUM_CLASSES, SEED,
    MODELS_DIR, FIGURES_DIR, METRICS_DIR,
    GRADCAM_LAYERS, SHAP_BACKGROUND_N,
    LIME_SAMPLES, LIME_FEATURES,
    IMG_SIZE_DEFAULT, IMG_SIZE_INCEPTION
)
from base_learner_training import PREPROCESS_FNS, IMG_SIZES


# ── Sample Image Selection ────────────────────────────────────

def get_sample_images(
        test_df    : pd.DataFrame,
        y_te       : np.ndarray,
        y_pred_ens : np.ndarray,
        y_prob_ens : np.ndarray
) -> dict:
    """
    Select one high-confidence correctly predicted image
    per class for XAI visualisation.

    Returns:
        Dict {class_name: {'path': str, 'conf': float}}
    """
    samples = {}
    for ci, cls in enumerate(CLASSES):
        correct  = (y_pred_ens == ci) & (y_te == ci)
        idx      = np.where(correct)[0]
        if len(idx) > 0:
            best_idx   = idx[
                np.argmax(y_prob_ens[idx, ci])]
            samples[cls] = {
                'path': test_df.iloc[best_idx]['filepath'],
                'conf': float(y_prob_ens[best_idx, ci]),
                'idx' : int(best_idx),
            }
    return samples


# ── Grad-CAM ──────────────────────────────────────────────────

def get_gradcam_heatmap(
        model      : keras.Model,
        img_array  : np.ndarray,
        layer_name : str,
        cls_idx    : int
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for a given class.

    Args:
        model     : Trained Keras model
        img_array : Preprocessed image (1, H, W, 3)
        layer_name: Target convolutional layer name
        cls_idx   : Class index to explain

    Returns:
        Heatmap array (224, 224) normalised to [0,1]
    """
    try:
        last_layer = model.get_layer(layer_name)
    except ValueError:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_layer = layer
                break

    grad_model = tf.keras.Model(
        model.inputs,
        [last_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_score     = preds[:, cls_idx]

    grads   = tape.gradient(class_score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0,1,2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / \
              (tf.math.reduce_max(heatmap) + 1e-8)
    hm_np   = heatmap.numpy()

    return cv2.resize(hm_np, (224, 224))


def overlay_heatmap(img_path : str,
                     heatmap  : np.ndarray,
                     alpha    : float = 0.45,
                     sz       : int = 224
                     ) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original X-ray."""
    img = cv2.cvtColor(
        cv2.resize(cv2.imread(img_path), (sz, sz)),
        cv2.COLOR_BGR2RGB)
    hm  = cv2.applyColorMap(
        np.uint8(255 * cv2.resize(heatmap, (sz, sz))),
        cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1-alpha, hm, alpha, 0)


def generate_gradcam_panel(
        sample_images: dict,
        save         : bool = True
) -> None:
    """
    Generate 4×6 Grad-CAM panel figure
    (4 classes × 6 columns: original + 5 models).
    """
    from tensorflow import keras as K
    MODELS_INPUT = (
        '/kaggle/input/datasets/mdkawshermahbub/'
        'pulmonary-model-outputs/models/')

    models_config = [
        ('DenseNet201',    PREPROCESS_FNS['DenseNet201'],    224),
        ('EfficientNetB4', PREPROCESS_FNS['EfficientNetB4'], 224),
        ('ResNet50V2',     PREPROCESS_FNS['ResNet50V2'],     224),
        ('InceptionV3',    PREPROCESS_FNS['InceptionV3'],    299),
        ('ConvNeXtTiny',   PREPROCESS_FNS['ConvNeXtTiny'],   224),
    ]

    os.makedirs(FIGURES_DIR+'gradcam', exist_ok=True)

    # Load all models
    loaded = {}
    for name, _, _ in models_config:
        loaded[name] = K.models.load_model(
            MODELS_INPUT + f'{name}_final.keras')

    fig, axes = plt.subplots(4, 6, figsize=(26, 18))
    fig.suptitle(
        'Grad-CAM Visualisations — 5 CNNs × 4 Classes',
        fontsize=13, fontweight='bold')

    for ri, cls in enumerate(CLASSES):
        if cls not in sample_images:
            continue
        img_path = sample_images[cls]['path']

        # Original
        orig = cv2.cvtColor(
            cv2.resize(cv2.imread(img_path), (224,224)),
            cv2.COLOR_BGR2RGB)
        axes[ri][0].imshow(orig)
        axes[ri][0].set_title(
            f'{cls}\n(Original)',
            fontweight='bold', fontsize=10)
        axes[ri][0].axis('off')

        # Per model
        for ci, (mname, pre_fn, img_sz) in \
                enumerate(models_config):
            model  = loaded[mname]
            img_pp = np.expand_dims(
                pre_fn(img_to_array(
                    load_img(img_path,
                             target_size=(img_sz,img_sz)))),
                axis=0)
            hm     = get_gradcam_heatmap(
                model, img_pp,
                GRADCAM_LAYERS[mname],
                CLASSES.index(cls))
            ov     = overlay_heatmap(img_path, hm)
            conf   = model.predict(
                img_pp, verbose=0)[0][CLASSES.index(cls)]

            axes[ri][ci+1].imshow(ov)
            axes[ri][ci+1].set_title(
                f'{mname}\n{conf:.3f}', fontsize=8)
            axes[ri][ci+1].axis('off')

    plt.tight_layout()
    if save:
        plt.savefig(
            FIGURES_DIR+'gradcam/Figure_GradCAM_Panel.png',
            dpi=300, bbox_inches='tight')
    plt.show()

    # Cleanup
    for m in loaded.values():
        del m
    import gc; gc.collect()
    tf.keras.backend.clear_session()


# ── SHAP ──────────────────────────────────────────────────────

def compute_shap_values(
        model        : keras.Model,
        test_imgs    : np.ndarray,
        background   : np.ndarray
) -> list:
    """
    Compute SHAP DeepExplainer values.

    Args:
        model      : DenseNet201 model
        test_imgs  : Test images (N, 224, 224, 3)
        background : Background reference (M, 224, 224, 3)

    Returns:
        List of SHAP value arrays per image
    """
    import shap
    explainer   = shap.DeepExplainer(model, background)
    shap_values = []

    for i, img in enumerate(test_imgs):
        print(f"  SHAP image {i+1}/{len(test_imgs)}...",
              end=' ')
        sv = explainer.shap_values(img[np.newaxis,...])
        shap_values.append(sv)
        print("✅")

    return shap_values


def get_shap_spatial_map(
        shap_values: list,
        img_idx    : int,
        cls_idx    : int
) -> np.ndarray:
    """
    Extract spatial importance map from SHAP values.
    Handles shape (1, 224, 224, 3, 4).

    Returns:
        Spatial map (224, 224)
    """
    sv = shap_values[img_idx]
    if isinstance(sv, list):
        return sv[cls_idx][0].sum(axis=2)
    elif sv.ndim == 5:
        # shape (1, H, W, 3, num_classes)
        return sv[0, :, :, :, cls_idx].sum(axis=2)
    else:
        return sv[0].sum(axis=-1)


def compute_shap_region_importance(
        shap_values   : list,
        test_labels   : list,
        h             : int = 224,
        w             : int = 224,
        save          : bool = True
) -> pd.DataFrame:
    """
    Compute percentage attribution per lung region.

    Regions: Upper-Left, Upper-Right,
             Lower-Left, Lower-Right, Centre

    Returns:
        DataFrame with region importance per class
    """
    n       = len(shap_values)
    regions = {
        'Upper-Left' : (0,    h//2, 0,    w//2),
        'Upper-Right': (0,    h//2, w//2, w),
        'Lower-Left' : (h//2, h,   0,    w//2),
        'Lower-Right': (h//2, h,   w//2, w),
        'Centre'     : (h//4, 3*h//4, w//4, 3*w//4),
    }

    rows = []
    for ci, cls in enumerate(CLASSES):
        sv_list = [
            np.abs(get_shap_spatial_map(
                shap_values, i, ci))
            for i in range(n)]
        mean_abs = np.array(sv_list).mean(axis=0)
        total    = mean_abs.sum()

        for reg_name, (r1,r2,c1,c2) in regions.items():
            pct = mean_abs[r1:r2, c1:c2].sum() / \
                  total * 100
            rows.append({
                'Class'     : cls,
                'Region'    : reg_name,
                'Importance': round(pct, 2),
            })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'shap_region_importance.csv',
            index=False)
    return df


# ── LIME ──────────────────────────────────────────────────────

def lime_predict_fn(model: keras.Model, pre_fn):
    """Return LIME-compatible predict function."""
    def predict(images: np.ndarray) -> np.ndarray:
        processed = np.array([
            pre_fn(img.astype(np.float32))
            for img in images])
        return model.predict(processed, verbose=0)
    return predict


def generate_lime_explanations(
        model        : keras.Model,
        sample_images: dict,
        save         : bool = True
) -> dict:
    """
    Generate LIME superpixel explanations for all classes.

    Returns:
        Dict {class_name: lime_explanation}
    """
    from lime import lime_image
    pre_fn      = PREPROCESS_FNS['DenseNet201']
    predict_fn  = lime_predict_fn(model, pre_fn)
    explainer   = lime_image.LimeImageExplainer(
        random_state=SEED)
    lime_results= {}

    os.makedirs(FIGURES_DIR+'lime', exist_ok=True)

    for cls in CLASSES:
        if cls not in sample_images:
            continue

        img_rgb = cv2.cvtColor(
            cv2.resize(
                cv2.imread(sample_images[cls]['path']),
                (224, 224)),
            cv2.COLOR_BGR2RGB)

        print(f"  LIME {cls}: {LIME_SAMPLES} "
              f"perturbations...", end=' ')
        explanation = explainer.explain_instance(
            img_rgb,
            predict_fn,
            top_labels  = NUM_CLASSES,
            hide_color  = 0,
            num_samples = LIME_SAMPLES,
            random_seed = SEED,
        )
        lime_results[cls] = {
            'explanation': explanation,
            'image'      : img_rgb,
            'conf'       : float(
                predict_fn(
                    img_rgb[np.newaxis,...])[0][
                    CLASSES.index(cls)]),
            'cls_idx'    : CLASSES.index(cls),
        }
        print("✅")

    return lime_results


def analyse_lime_results(
        lime_results: dict,
        save        : bool = True
) -> pd.DataFrame:
    """
    Extract quantitative LIME superpixel statistics.

    Returns:
        DataFrame with per-class LIME metrics
    """
    rows = []
    print("\nLIME Quantitative Analysis:")
    print("=" * 50)

    for cls in CLASSES:
        if cls not in lime_results:
            continue

        res       = lime_results[cls]
        cls_idx   = res['cls_idx']
        local_exp = res['explanation'].local_exp.get(
            cls_idx, [])

        if local_exp:
            weights     = [w for _, w in local_exp]
            pos_weights = [w for w in weights if w > 0]
            neg_weights = [w for w in weights if w < 0]

            print(f"\n  {cls}:")
            print(f"    Total segments  : {len(weights)}")
            print(f"    Positive regions: {len(pos_weights)}")
            print(f"    Negative regions: {len(neg_weights)}")
            if pos_weights:
                print(f"    Max pos weight  : "
                      f"{max(pos_weights):.4f}")
                print(f"    Mean pos weight : "
                      f"{np.mean(pos_weights):.4f}")

            rows.append({
                'Class'           : cls,
                'Confidence'      : round(res['conf'], 4),
                'Total_segments'  : len(weights),
                'Positive_regions': len(pos_weights),
                'Negative_regions': len(neg_weights),
                'Max_pos_weight'  : round(
                    max(pos_weights), 4)
                    if pos_weights else 0,
                'Mean_pos_weight' : round(
                    np.mean(pos_weights), 4)
                    if pos_weights else 0,
            })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'lime_analysis.csv', index=False)
    return df


# ── XAI Agreement ─────────────────────────────────────────────

def compute_xai_agreement(
        gradcam_maps : dict,
        shap_maps    : dict,
        save         : bool = True
) -> pd.DataFrame:
    """
    Compute spatial correlation between Grad-CAM
    and SHAP attribution maps.

    Metrics:
        - Pixel-level Spearman r
        - Pixel-level Pearson r
        - Region-level Spearman r (4×4 grid)

    Returns:
        DataFrame with agreement metrics per class
    """
    rows = []
    print("\nXAI Agreement Analysis (Grad-CAM vs SHAP):")
    print("=" * 55)

    for cls in CLASSES:
        if cls not in gradcam_maps or \
           cls not in shap_maps:
            continue

        gc_map   = gradcam_maps[cls]
        shap_map = shap_maps[cls]

        # Normalise to [0,1]
        gc_norm   = (gc_map  -gc_map.min()) / \
                    (gc_map.max()  -gc_map.min()  +1e-8)
        shap_norm = (shap_map-shap_map.min()) / \
                    (shap_map.max()-shap_map.min()+1e-8)

        # Pixel correlation
        spear_r, spear_p = spearmanr(
            gc_norm.flatten(), shap_norm.flatten())
        pear_r,  pear_p  = pearsonr(
            gc_norm.flatten(), shap_norm.flatten())

        # Region-level (4×4 grid)
        grid      = 56
        gc_reg    = []
        shap_reg  = []
        for ri in range(4):
            for ci2 in range(4):
                r1, r2 = ri*grid,  (ri+1)*grid
                c1, c2 = ci2*grid, (ci2+1)*grid
                gc_reg.append(
                    gc_norm[r1:r2, c1:c2].mean())
                shap_reg.append(
                    shap_norm[r1:r2, c1:c2].mean())

        reg_r, _ = spearmanr(gc_reg, shap_reg)

        # Interpretation
        if   spear_r > 0.5: agreement = "Strong"
        elif spear_r > 0.3: agreement = "Moderate"
        elif spear_r > 0.1: agreement = "Weak"
        elif spear_r > 0.0: agreement = "Minimal"
        else:               agreement = "No agreement"

        print(f"\n  {cls}:")
        print(f"    Pixel Spearman r : {spear_r:.4f}")
        print(f"    Pixel Pearson  r : {pear_r:.4f}")
        print(f"    Region Spearman r: {reg_r:.4f}")
        print(f"    Interpretation   : {agreement}")

        rows.append({
            'Class'           : cls,
            'Spearman_r'      : round(spear_r, 4),
            'Spearman_p'      : round(spear_p, 4),
            'Pearson_r'       : round(pear_r,  4),
            'Region_Spearman' : round(reg_r,   4),
            'Agreement'       : agreement,
        })

    df = pd.DataFrame(rows)
    if save:
        df.to_csv(
            METRICS_DIR+'xai_agreement.csv', index=False)
    return df


if __name__ == '__main__':
    print("XAI analysis module loaded.")
    print(f"Methods: Grad-CAM, SHAP, LIME")
    print(f"SHAP background: {SHAP_BACKGROUND_N} images")
    print(f"LIME samples   : {LIME_SAMPLES}")
