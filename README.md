# ADetailer

ADetailer is an extension for the stable diffusion webui that does automatic masking and inpainting. It is similar to the Detection Detailer.

## Install

You can install it directly from the Extensions tab.

![image](https://i.imgur.com/qaXtoI6.png)

Or

(from Mikubill/sd-webui-controlnet)

1. Open "Extensions" tab.
2. Open "Install from URL" tab in the tab.
3. Enter `https://github.com/Bing-su/adetailer.git` to "URL for extension's git repository".
4. Press "Install" button.
5. Wait 5 seconds, and you will see the message "Installed into stable-diffusion-webui\extensions\adetailer. Use Installed tab to restart".
6. Go to "Installed" tab, click "Check for updates", and then click "Apply and restart UI". (The next time you can also use this method to update extensions.)
7. Completely restart A1111 webui including your terminal. (If you do not know what is a "terminal", you can reboot your computer: turn your computer off and turn it on again.)

## Options

| Model, Prompts                    |                                                                                    |                                                                                                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADetailer model                   | Determine what to detect.                                                          | `None` = disable                                                                                                                                       |
| ADetailer model classes           | Comma separated class names to detect. only available when using YOLO World models | If blank, use default values.<br/>default = [COCO 80 classes](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) |
| ADetailer prompt, negative prompt | Prompts and negative prompts to apply                                              | If left blank, it will use the same as the input.                                                                                                      |
| Skip img2img                      | Skip img2img. In practice, this works by changing the step count of img2img to 1.  | img2img only                                                                                                                                           |

| Detection                            |                                                                                              |              |
| ------------------------------------ | -------------------------------------------------------------------------------------------- | ------------ |
| Detection model confidence threshold | Only objects with a detection model confidence above this threshold are used for inpainting. |              |
| Mask min/max ratio                   | Only use masks whose area is between those ratios for the area of the entire image.          |              |
| Mask only the top k largest          | Only use the k objects with the largest area of the bbox.                                    | 0 to disable |

If you want to exclude objects in the background, try setting the min ratio to around `0.01`.

| Mask Preprocessing              |                                                                                                                                     |                                                                                         |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Mask x, y offset                | Moves the mask horizontally and vertically by                                                                                       |                                                                                         |
| Mask erosion (-) / dilation (+) | Enlarge or reduce the detected mask.                                                                                                | [opencv example](https://docs.opencv.org/4.7.0/db/df6/tutorial_erosion_dilatation.html) |
| Mask merge mode                 | `None`: Inpaint each mask<br/>`Merge`: Merge all masks and inpaint<br/>`Merge and Invert`: Merge all masks and Invert, then inpaint |                                                                                         |

Applied in this order: x, y offset → erosion/dilation → merge/invert.

### Face filter (fork addition)

Each ADetailer tab has a **Face filter** dropdown: `Any`, `Female`, `Male`, `Female 1`, `Female 2`, `Female 3`, `Male 1`, `Male 2`, `Male 3`.

- `Any` (default) is the original behaviour: every detected object is inpainted.
- `Female` / `Male` classify each detected box with CLIP zero-shot (the crop is taken with a 25% margin and scored against an ensemble of three female and three male prompts) and only inpaint the faces whose label matches.
- `Female 1` … `Male 3` additionally pick out a single face by position: `Female 2` is the **second** female face in the frame.

Because each tab has its own prompt, setting tab 1 to `Female` and tab 2 to `Male` gives you a separate inpaint pass per gender, regardless of where each person stands in the frame. With the ordinal choices you can go further and give each woman in a group shot her own prompt: tab 1 `Female 1`, tab 2 `Female 2`, and so on.

**Ordering.** The ordinal counts faces in the bounding box sort order set in `Settings -> ADetailer -> bounding box sort` (`Position (left to right)` by default, so `Female 2` is the second woman from the left). Only faces of the tab's own gender are counted: in a row of woman, man, woman, the second woman is `Female 2`, not `Female 3`. The ordinal is applied right after the gender split and *before* the tab's mask ratio / top-k filters, so `Female 2` always means the second woman in the frame and never "the second-largest mask". If fewer faces of that gender were found than the ordinal asks for, the tab inpaints nothing, exactly as when no face matched at all.

**Unknown faces.** A face whose winning probability is below the confidence threshold counts as *unknown*. Unknown faces are never silently dropped from the whole run:

- if any other enabled tab has its face filter set to `Any`, unknown faces are skipped here, because that tab will inpaint them anyway;
- otherwise they are handed to the lowest-index enabled tab that has a filter set, so exactly one tab keeps them.

An unknown face routed to a tab this way counts as a member of that tab's gender for ordinal purposes, i.e. it takes up a place in the tab's `Female 1 / 2 / 3` numbering.

If the classifier cannot be loaded or fails, a warning is printed and the tab processes all detected faces as usual.

Two settings live in `Settings -> ADetailer`:

| Setting                                   | Default                        |                                                                        |
| ----------------------------------------- | ------------------------------ | ---------------------------------------------------------------------- |
| Face filter: minimum classifier confidence | `0.6`                          | Faces scoring below this count as unknown.                             |
| Face filter: CLIP model name               | `openai/clip-vit-large-patch14` | Any Hugging Face CLIP id. Downloaded to the usual Hugging Face cache. |

#### Inpainting

Each option corresponds to a corresponding option on the inpaint tab. Therefore, please refer to the inpaint tab for usage details on how to use each option.

## ControlNet Inpainting

You can use the ControlNet extension if you have ControlNet installed and ControlNet models.

Support `inpaint, scribble, lineart, openpose, tile, depth` controlnet models. Once you choose a model, the preprocessor is set automatically. It works separately from the model set by the Controlnet extension.

If you select `Passthrough`, the controlnet settings you set outside of ADetailer will be used.

## Advanced Options

API request example: [wiki/REST-API](https://github.com/Bing-su/adetailer/wiki/REST-API)

`[SEP], [SKIP], [PROMPT]` tokens: [wiki/Advanced](https://github.com/Bing-su/adetailer/wiki/Advanced)

## Media

- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ① 【Stable Diffusion】](https://youtu.be/sF3POwPUWCE)
- 🎥 [どこよりも詳しい After Detailer (adetailer)の使い方 ② 【Stable Diffusion】](https://youtu.be/urNISRdbIEg)

- 📜 [ADetailer Installation and 5 Usage Methods](https://kindanai.com/en/manual-adetailer/)

## Model

| Model                 | Target                | mAP 50                        | mAP 50-95                     |
| --------------------- | --------------------- | ----------------------------- | ----------------------------- |
| face_yolov8n.pt       | 2D / realistic face   | 0.660                         | 0.366                         |
| face_yolov8s.pt       | 2D / realistic face   | 0.713                         | 0.404                         |
| hand_yolov8n.pt       | 2D / realistic hand   | 0.767                         | 0.505                         |
| person_yolov8n-seg.pt | 2D / realistic person | 0.782 (bbox)<br/>0.761 (mask) | 0.555 (bbox)<br/>0.460 (mask) |
| person_yolov8s-seg.pt | 2D / realistic person | 0.824 (bbox)<br/>0.809 (mask) | 0.605 (bbox)<br/>0.508 (mask) |
| mediapipe_face_full   | realistic face        | -                             | -                             |
| mediapipe_face_short  | realistic face        | -                             | -                             |
| mediapipe_face_mesh   | realistic face        | -                             | -                             |

The YOLO models can be found on huggingface [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer).

For a detailed description of the YOLO8 model, see: https://docs.ultralytics.com/models/yolov8/#overview

YOLO World model: https://docs.ultralytics.com/models/yolo-world/

### Additional Model

Put your [ultralytics](https://github.com/ultralytics/ultralytics) yolo model in `models/adetailer`. The model name should end with `.pt`.

It must be a bbox detection or segment model and use all label.

## How it works

ADetailer works in three simple steps.

1. Create an image.
2. Detect object with a detection model and create a mask image.
3. Inpaint using the image from 1 and the mask from 2.

## Development

ADetailer is developed and tested using the stable-diffusion 1.5 model, for the latest version of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) repository only.

## License

ADetailer is a derivative work that uses two AGPL-licensed works (stable-diffusion-webui, ultralytics) and is therefore distributed under the AGPL license.

## See Also

- https://github.com/ototadana/sd-face-editor
- https://github.com/continue-revolution/sd-webui-segment-anything
- https://github.com/portu-sim/sd-webui-bmab
