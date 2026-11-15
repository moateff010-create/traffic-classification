import tensorflow as tf
from keras import layers,models
IMG_HEIGHT = 32
IMG_WIDTH = 32
BATCH_SIZE = 32
NUM_CLASSES = 43
train_df = tf.keras.utils.image_dataset_from_directory(
    r"D:/datasets/Train",
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    r"D:/datasets/Train",
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)
AUTOTUNE = tf.data.AUTOTUNE
train_df = train_df.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
model = models.Sequential([
    layers.Rescaling(1./255,input_shape = (IMG_HEIGHT,IMG_WIDTH,3)),
    layers.Conv2D(32,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(64,(3,3),activation="relu"),
    layers.MaxPool2D((2,2)),
    layers.Flatten(),
    layers.Dense(128,activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES,activation="softmax")
])

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
model.summary()

history = model.fit(
    train_df,
    shuffle=False,
    validation_data=val_ds,
    epochs = 10
)
test_df = tf.keras.utils.image_dataset_from_directory(
    r"D:/datasets/Test",
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)
test_loss,test_accuracy = model.evaluate(test_df)
print(f"test accuracy {test_accuracy*100:.2f}")
