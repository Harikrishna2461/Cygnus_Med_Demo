**BASIC INFORMATION :** 

A ready-to-run folder that finds the fascia layer and vein cross-sections in

peripheral vascular ultrasound images/video, using two finetuned AI models

(not the generic/stock BioMedParse model). Everything needed is inside this

one folder - checkpoints, code, config. You do not need any other project

folder.



**STEP 1 - INSTALL DEPENDENCIES :**

Open a terminal/command prompt INSIDE this folder and run:



"pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128"



Note the "--extra-index-url" part is required - it is what lets pip find the

correct GPU (CUDA) version of PyTorch. Leaving it out will cause an error.



This step downloads a few GB of packages (PyTorch itself is large) and can

take several minutes depending on your internet connection.



If `nvidia-smi` showed a CUDA version lower than 12.8, open requirements.txt

in a text editor, change "+cu128" to match (e.g. "+cu121"), also change

"cu128" to the same value in the pip command above, then re-run it.



**STEP 2 - TRY IT ON ONE IMAGE FIRST (recommended sanity check) :**

Before running a whole video, confirm everything works on a single ultrasound

frame/screenshot:



&#x20;   python segment\_image.py --input frame.png --output frame\_annotated.png



\- Replace frame.png with the path to any real ultrasound image you have.

\- The first run is slow (30-60+ seconds) because it has to load both models

&#x20; onto the GPU. This is normal.

\- It will print how many vein cross-sections it found, and save

&#x20; frame\_annotated.png with:

&#x20;   \* GREEN outlines = detected veins, labeled V1, V2, ...

&#x20;   \* AMBER/ORANGE lines = fascia layer (superficial + deep boundary)

\- Open frame\_annotated.png and check it looks reasonable before moving on.



**STEP 3 - RUN ON A FULL VIDEO :**

&#x20; Run this command in terminal in root folder :

&#x09;python annotate\_video.py --input scan.mp4 --output scan\_annotated.mp4



\- Replace scan.mp4 with your ultrasound video file.

\- This processes every frame and writes a new video (scan\_annotated.mp4)

&#x20; with the same green/amber overlays as above, playable in a normal browser

&#x20; or video player.

\- Progress is printed to the terminal every 50 frames along with a rough

&#x20; frames-per-second speed estimate.

\- For a long video, if you want it to run faster (at the cost of slightly

&#x20; less smooth annotations), add --every N, e.g.:



&#x20;   python annotate\_video.py --input scan.mp4 --output scan\_annotated.mp4 --every 3



&#x20; This only runs the AI models on every 3rd frame and reuses the previous

&#x20; frame's overlay in between, roughly 3x faster.



**USING IT FROM YOUR OWN PYTHON CODE :**

If you want to call the segmentation from your own script instead of the

command-line tools above:



&#x20;   import cv2

&#x20;   import engine



&#x20;   frame = cv2.imread("frame.png")          # any OpenCV BGR image/frame

&#x20;   blobs, fascia = engine.segment\_frame(frame)



&#x20;   print(f"{len(blobs)} vein(s) found")

&#x20;   for b in blobs:

&#x20;       print(b.blob\_id, b.centroid, b.area\_px)   # centroid = (x, y) in pixels



&#x20;   # fascia.sup\_row\_at\_col / fascia.deep\_row\_at\_col are arrays giving the

&#x20;   # fascia's row (y) position at every column (x) across the image width,

&#x20;   # with NaN where the fascia wasn't confidently detected at that column.



The first call to engine.segment\_frame() loads both models (slow, one-time);

every call after that reuses them and is much faster.



**COMMON PROBLEMS :**

\- "CUDA out of memory" -> close other GPU-heavy programs

\- "Checkpoint not found" error -> check the file path and you might have to change to the absolute path .

\- pip install fails on torch/torchvision -> you likely forgot the

&#x20; --extra-index-url flag in STEP 1, or your driver doesn't support CUDA 12.8

&#x20; (see STEP 1's note about checking with nvidia-smi).

\- A warning about "MultiScaleDeformableAttention CUDA op not found" when you

&#x20; first run anything -> this is expected and harmless, safe to ignore.



**WHAT'S IN THIS FOLDER :**

&#x20;   checkpoints\\fascia\\model\_state\_dict.pt   - fascia AI model (\~1.7 GB)

&#x20;   checkpoints\\vein\\model\_state\_dict.pt     - vein AI model (\~1.7 GB)

&#x20;   configs\\biomed\_fascia\_finetuning.yaml    - model architecture settings

&#x20;   modeling\\, utilities\\, stubs\\            - supporting code, do not edit

&#x20;   config.py            - tweakable settings (sensitivity thresholds, colors, etc.)

&#x20;   engine.py             - core segmentation logic

&#x20;   annotate\_video.py     - command-line tool for videos (STEP 4 above)

&#x20;   segment\_image.py      - command-line tool for single images (STEP 3 above)

&#x20;   requirements.txt      - Python package list (STEP 1 above)

&#x20;   USAGE\_GUIDE.txt        - this file



**WHY TWO SEPARATE MODEL FILES :**

One model (checkpoints\\fascia) was trained to find the fascia layer, the

other (checkpoints\\vein) was trained to find vein cross-sections. They are

kept separate because a single combined model performed noticeably worse at

finding the fascia layer during testing. Both are used automatically - you

never need to choose between them.



