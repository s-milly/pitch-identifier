# What pitch was that?

A pitch classifier trained on every MLB pitch from the 2024 and 2025 regular
seasons. Give it a pitch's velocity, spin rate, induced vertical break,
horizontal break and spin direction, plus the pitcher's fastball, and it names
the pitch: four-seam, sinker, cutter, slider, sweeper, curveball, changeup or
splitter.

**Live page:** https://s-milly.github.io/pitch-identifier/

## Results

| Test | Accuracy |
|---|---|
| MLB pitchers the model never saw, one pitch at a time | 86.8% |
| MLB pitchers the model never saw, whole outing | 87.8% |
| College pitchers (TrackMan, hand-tagged, 5 pitch groups) | 90.1% |
| TrackMan's own automatic tags on the same college pitches | 78.0% |

MLB scores come from 5-fold cross-validation split by pitcher, so every pitch
is graded by a model that never trained on that pitcher.

## How it works

- **Features.** Velocity, spin, induced vertical break, arm-side horizontal
  break and spin direction, plus each pitch's gap from the pitcher's own
  fastball (the median of the hardest 15% of their pitches). Lefties are
  mirrored onto the right-handed scale.
- **Model.** LightGBM, 150 rounds, 15 leaves per tree. The trees are exported
  to `site/model.json` and evaluated in the browser by a short tree walk.
- **Smoothing.** When a whole outing is available, each pitch's probabilities
  are averaged with its 10 most similar pitches from the same pitcher.

## Run it

```bash
pip install pybaseball lightgbm scikit-learn pandas pyarrow
python scripts/fetch_statcast.py 2024
python scripts/fetch_statcast.py 2025
python scripts/prepare_statcast.py 2024 2025
python scripts/train.py            # writes data/metrics.json and site/model.json
python scripts/college.py          # downloads the college data and scores it
python scripts/build_page_data.py  # writes site/page_data.json for the page
python -m http.server              # then open http://localhost:8000
```

To run a Rapsodo pitching export through the model:

```bash
python scripts/rapsodo.py path/to/session.csv --throws R
```

## Data

- MLB pitch data from [Baseball Savant](https://baseballsavant.mlb.com/), pulled
  with [pybaseball](https://github.com/jldbc/pybaseball).
- College data: Pifer, N. D., *Optical Tracking Data from College Baseball
  Scrimmages*, Mendeley Data, V3,
  [doi:10.17632/xfnz6mkdzm.3](https://doi.org/10.17632/xfnz6mkdzm.3), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes made: pitch
  types were grouped into the dataset's five hand-tag groups. The data is
  downloaded by `scripts/college.py` and is not stored in this repository.
