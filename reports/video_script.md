# Presentation video: shot list and narration (about 5 minutes)

Every number below comes from the README or report. Say these numbers and no others.

## Before recording

1. Start the API once so the model is loaded (it takes about a minute to start):
   `uv run uvicorn src.api.main:app --port 8000`
2. Open in browser tabs: `http://localhost:8000/docs`, the Hugging Face model repo, the Hugging Face Space.
   Make the repos public first, or record while logged in.
3. Pre-run `curl "localhost:8000/predict/random-validation?seed=3"` so you know the output looks right.
4. Terminal font 18pt+, browser zoom 125%.

## Shots

| # | Time | On screen | Narration |
|---|------|-----------|-----------|
| 1 | 0:00–0:25 | `plots/videoAnnot.png` or `reports/figures/output_sample.gif` | "In volleyball, the question isn't just what one player is doing, it's what the whole team is doing: a left-side set, a right-side spike, a winning point. This project recognizes 8 group activities from 9 frames of video, using the Volleyball dataset and the 2016 CVPR paper by Ibrahim et al." |
| 2 | 0:25–0:55 | README "At a Glance": dataset line | "55 videos, 4,830 clips, labels at two levels: the group activity, and what each player is doing. I built nine baselines on one shared data loader, trainer and evaluator, so every comparison is fair." |
| 3 | 0:55–2:00 | `reports/figures/progression.png` (hold on it) | "Each model adds one idea. A single frame gets 62.6%. Adding player crops, then LSTMs over time, climbs to 73.8% with a hierarchy: a player LSTM, pooling, then a scene LSTM. But that model confuses left and right: it pools all players together, so it can't tell which team is acting. Pooling each team separately, B8, jumps 11.9 points to 85.6%, above the paper's 81.9%." |
| 4 | 2:00–2:20 | `reports/figures/b8_confusion_matrix.png` | "The confusion matrix shows the left/right confusion is largely gone." (Only say "largely" if the matrix shows it; look at it first.) |
| 5 | 2:20–3:10 | README "B1–B8 vs. B9" table, then the B9 confusion matrix | "As a control I threw the hierarchy away. B9 is one YOLO classifier on the raw frame: no player boxes, no player labels. Scored per frame it got 78.9%. Scored per clip, with a vote over each clip's ten frames, so it's comparable to the others, it gets 80.3%. So the hierarchy buys about 5 points, and B9's errors are set versus pass: one player's arm posture, about 17 by 62 pixels in a 224-pixel frame." |
| 6 | 3:10–4:10 | Terminal + `/docs` | "Now the demo. The service loads B8 once, then serves it. This endpoint picks a random held-out validation clip and returns the prediction, the 8 probabilities and the player boxes." Run `curl "localhost:8000/predict/random-validation?seed=3"`. Then open `/predict/random-validation/video?seed=3` in the browser: the GIF shows boxes coloured by team and predicted vs. true label. "Blue is the left team, orange is the right. The banner is green when the prediction matches." Run it with 2 or 3 more seeds; if one is wrong, say so. |
| 7 | 4:10–4:35 | Hugging Face Space and model repo | "Weights and a model card are on Hugging Face, and the demo Space shows precomputed predictions on 20 held-out clips. It's static, not live, because free Gradio hosting needs a paid plan." |
| 8 | 4:35–5:00 | README limitations / report "Next Steps" | "Limits: the model needs tracked player boxes, taken here from the dataset annotations. There's no detector or tracker, so accuracy on new video is unmeasured. Next: a detector plus tracker in front of B8, and measuring that drop." |

## Things not to claim

- Don't say 82% (the 200-clip sanity check) or 20/20 (the demo clips) is the model's accuracy. The test accuracy is 85.64%.
- Don't say the demo works on arbitrary video. It works on dataset clips that come with boxes.
- Don't call the Space "live". It shows stored results.
- B9's best run was 83.17%, but that checkpoint no longer exists, so only the 80.3% / 78.9% figures are reproducible.

## Recording tips

- Record at 1080p, one take per shot, and cut together; a 5-minute run is realistic with 8 clips.
- On shot 6, keep the first seed that works and any wrong prediction you get. A wrong example makes the demo more credible.
- OBS or `wf-recorder` (Wayland/Fedora) are fine; add the narration afterwards if you prefer.
