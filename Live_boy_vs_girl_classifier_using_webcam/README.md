# Boy vs Girl Classifier (CNN, Google Colab)

A small image classification project that captures webcam photos, trains a CNN to
distinguish between two classes ("boy" / "girl"), and runs live predictions from
the webcam.

> ⚠️ This notebook is designed to run in **Google Colab**, since it relies on
> Colab-specific JS/webcam and file-download utilities (`google.colab.output`,
> `google.colab.files`). It will not run as-is on a local Jupyter install.

## What this notebook does

1. **Webcam capture** — Opens your browser's webcam and saves a series of frames
   (`collect_images`) to a local folder for each class.
2. **Dataset zipping** — Zips the captured images and downloads them to your
   computer (`boy_images.zip`, `girl_images.zip`).
3. **Preprocessing** — Loads images, resizes to 100x100, normalizes pixel values,
   and splits into train/test sets.
4. **Model training** — Trains a simple Conv2D CNN (Keras/TensorFlow) for binary
   classification.
5. **Model export** — Saves and downloads the trained model (`boy_girl_classifier.h5`).
6. **Live inference** — Captures new webcam snapshots and classifies them in
   real time, displaying label + confidence.

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run the setup cells (imports, webcam helper functions) in order.
3. Run the **boy** image collection cell, then the **girl** image collection cell.
   Each captures 50 photos with a short interval between shots — make sure
   only the intended subject is in frame.
4. Run the training cells to build and train the CNN.
5. Run the live classification cell to test the model on new webcam frames.

## Data & privacy notes

- All captured images are stored **only on the temporary Colab VM disk**
  (`/content/...`) for the duration of your session. They are **not** uploaded
  anywhere automatically and are **not** embedded in the saved `.ipynb` file.
- The Colab VM and all files on it are wiped when the runtime disconnects or
  is reset — so the dataset does **not** persist between sessions unless you
  explicitly download it.
- The notebook does push two things to your *own* machine via the browser
  download dialog: the zipped image folders and the trained `.h5` model.
  These stay local to whoever runs the notebook and downloads them.
- If you plan to share this notebook (e.g. on GitHub or with collaborators),
  make sure to **clear cell outputs** before sharing and do **not** attach the
  zip files or the `.h5` model unless you intend for others to have your
  training images.
- To remove all captured images and exported files from the Colab VM once
  you're done, run:

  ```python
  import shutil, os
  shutil.rmtree('/content/people-classifier', ignore_errors=True)
  for f in ['/content/boy_images.zip', '/content/girl_images.zip',
            '/content/boy_girl_classifier.h5']:
      if os.path.exists(f):
          os.remove(f)
  ```

## Requirements

- Google Colab environment (uses `google.colab` modules)
- Python packages (pre-installed in Colab): `tensorflow`, `numpy`, `pillow`,
  `scikit-learn`, `matplotlib`

## Limitations

- Small dataset (50 images per class by default) — accuracy and generalization
  will be limited; consider collecting more varied images (lighting, angle,
  background) for better results.
- Binary "boy"/"girl" framing is a simplification and may not generalize well
  or be appropriate outside of a personal learning exercise.
- The model is trained on whoever's face is captured during collection, so it
  will likely only recognize patterns similar to those specific images, not
  people in general.
