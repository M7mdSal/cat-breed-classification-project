from pathlib import Path
import argparse, json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description='Train a cat breed image classifier')
    parser.add_argument('--data', type=Path, default=ROOT/'data')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=160)
    parser.add_argument('--fine-tune', action='store_true', help='Fine-tune last 20 backbone layers after head training')
    args = parser.parse_args()
    tf.keras.utils.set_random_seed(42)
    breed_dirs = sorted(p for p in args.data.iterdir() if p.is_dir() and any(p.glob('*.*')))
    if len(breed_dirs) < 2:
        raise SystemExit('Add JPG/PNG images to at least two breed folders in data/, e.g. data/Persian/ and data/Siamese/.')
    common = dict(directory=str(args.data), validation_split=0.2, seed=42, image_size=(args.image_size,args.image_size), batch_size=args.batch_size, label_mode='int')
    train = tf.keras.utils.image_dataset_from_directory(subset='training', shuffle=True, **common)
    valid = tf.keras.utils.image_dataset_from_directory(subset='validation', shuffle=True, **common)
    labels = train.class_names
    if labels != valid.class_names: raise ValueError('Class labels do not match')
    autotune=tf.data.AUTOTUNE
    train=train.prefetch(autotune)
    valid=valid.prefetch(autotune)
    augmentation=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),tf.keras.layers.RandomRotation(.08),tf.keras.layers.RandomZoom(.1)],name='augmentation')
    backbone=tf.keras.applications.MobileNetV2(input_shape=(args.image_size,args.image_size,3),include_top=False,weights='imagenet')
    backbone.trainable=False
    inputs=tf.keras.Input(shape=(args.image_size,args.image_size,3))
    x=augmentation(inputs)
    x=tf.keras.layers.Rescaling(1./127.5,offset=-1)(x)
    x=backbone(x,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(.25)(x)
    outputs=tf.keras.layers.Dense(len(labels),activation='softmax')(x)
    model=tf.keras.Model(inputs,outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'outputs').mkdir(exist_ok=True)
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True,monitor='val_loss')]
    history=model.fit(train,validation_data=valid,epochs=args.epochs,callbacks=callbacks)
    if args.fine_tune:
        backbone.trainable=True
        for layer in backbone.layers[:-20]: layer.trainable=False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        extra=model.fit(train,validation_data=valid,epochs=5,callbacks=callbacks)
        for k in history.history: history.history[k].extend(extra.history[k])
    model.save(ROOT/'models'/'cat_breeds.keras')
    (ROOT/'models'/'labels.json').write_text(json.dumps(labels,indent=2),encoding='utf-8')
    loss,accuracy=model.evaluate(valid,verbose=0)
    (ROOT/'outputs'/'metrics.json').write_text(json.dumps({'validation_accuracy':float(accuracy),'validation_loss':float(loss),'classes':labels},indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,['accuracy','loss']):
        ax.plot(history.history[key],label='Train')
        ax.plot(history.history['val_'+key],label='Validation')
        ax.set_title(key.title()); ax.set_xlabel('Epoch'); ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'outputs'/'training_curves.png');plt.close(fig)
    print('Saved model, labels, validation metrics, and curves. Validation is not a held-out test set.')

if __name__=='__main__':main()
