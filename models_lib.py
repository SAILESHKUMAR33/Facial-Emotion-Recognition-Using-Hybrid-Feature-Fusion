"""Model builders: custom CNN, MobileNetV2 transfer model, landmark MLP, fusion model."""
import tensorflow as tf
from tensorflow.keras import Model, layers

from config import FEATURE_DIM, IMG_SIZE, MODELS_DIR


def build_custom_cnn(n_classes):
    """Baseline CNN from scratch: [Conv-BN-ReLU] x2 -> Pool -> Dropout, 4 blocks."""
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 255)(inp)
    for f in (32, 64, 128, 256):
        for _ in range(2):
            x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.25)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    return Model(inp, out, name="custom_cnn")


def build_transfer(n_classes):
    """MobileNetV2 (ImageNet) as a frozen feature extractor + new classifier."""
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inp)          # MobileNetV2 expects [-1, 1]
    mnet = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base = Model(mnet.input, mnet.output, name="mobilenetv2_base")   # named wrapper
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    return Model(inp, out, name="transfer_mobilenetv2")


def unfreeze_base_top(base, n):
    """Unfreeze the last n layers of the pretrained base (BatchNorm stays frozen)."""
    base.trainable = True
    for l in base.layers[:-n]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False


def build_landmark_mlp(input_dim, n_classes):
    inp = layers.Input((input_dim,), name="landmarks")
    x = layers.Dense(256)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    return Model(inp, out, name="landmark_mlp")


_BRANCH_FILES = {"cnn": "custom_cnn", "tl": "transfer_learning", "lm": "landmark_mlp"}


def _load_branch(key):
    m = tf.keras.models.load_model(MODELS_DIR / f"{_BRANCH_FILES[key]}.keras", compile=False)
    return Model(m.input, m.get_layer("features").output, name=f"{key}_branch")


def build_fusion(n_classes, branches, name="fusion"):
    """Feature-level fusion. branches is any subset of ('cnn', 'tl', 'lm').
    Each trained branch is loaded, frozen, its 'features' vector is batch-normalised,
    all vectors are concatenated, then Dense -> Dropout -> softmax."""
    inputs, feats = {}, []
    for key in branches:
        sub = _load_branch(key)
        sub.trainable = False
        if key == "lm":
            inputs.setdefault("landmarks", layers.Input(sub.input_shape[1:], name="landmarks"))
            x_in = inputs["landmarks"]
        else:
            inputs.setdefault("image", layers.Input((IMG_SIZE, IMG_SIZE, 3), name="image"))
            x_in = inputs["image"]
        f = sub(x_in, training=False)
        feats.append(layers.BatchNormalization(name=f"{key}_feat_bn")(f))
    x = feats[0] if len(feats) == 1 else layers.Concatenate(name="fusion_concat")(feats)
    x = layers.Dense(256, activation="relu", name="fusion_dense")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(x)
    return Model(list(inputs.values()), out, name=name)


def unfreeze_fusion_tl_top(fusion, n):
    branch = fusion.get_layer("tl_branch")
    branch.trainable = True
    unfreeze_base_top(branch.get_layer("mobilenetv2_base"), n)
